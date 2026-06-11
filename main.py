import argparse
import logging
import os

import av
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch._prims_common import DeviceLikeType
from ultralytics import YOLO
from migan_inference import Generator as MIGAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Программа для маскировки вотермарок на видео",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Настройки io
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Путь к входному видеофайлу",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Путь для выходного видеофайла.",
    )

    # Настройки программы
    parser.add_argument(
        "--mode",
        required=False,
        type=str,
        choices=["static", "dynamic"],
        default="dynamic",
        help="static — вотермарка неподвижна в пределах сцены (маска кэшируется), dynamic — может двигаться или появляться/исчезать внутри сцены",
    )
    parser.add_argument(
        "--skip_empty_scenes",
        action="store_true",
        help="Пропускать сцены, в которых не обнаружена вотермарка.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Размер батча для пакетной обработки кадров",
    )

    # Настройки scenedetect
    # Настройки yolo
    parser.add_argument(
        "--path_to_yolo_model",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство для вычислений (cuda или cpu)",
    )
    # Настройки video
    # Настройки migan
    parser.add_argument(
        "--path_to_migan_model",
        required=True,
        type=str,
    )
    # Настройки маски
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("x1", "y1", "x2", "y2"),
        help="Координаты вотермарки: левый верхний и правый нижний угол",
    )

    parser.add_argument(
        "--blur",
        type=int,
        default=0,
        required=False,
        help="Сила размытия маски",
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        required=False,
        help="Внешний отступ для выделенной области.",  # Внешний отступ делать пропорциональным или строгим (проценты или пиксели?)
    )

    # Параметры для предварительного просмотра работы программы
    parser.add_argument(
        "--mask-preview",
        action="store_true",
        help="При включении сохраняет маску для первого кадра видео в выходную директорию.",
    )

    parser.add_argument(
        "--inpaint-preview",
        action="store_true",
        help="При включении обрабатывает первый кадр видео и сохраняет его в выходную директорию.",
    )
    return parser


def detect_scenes():
    pass


def expand_bbox(bbox, padding: int, width: int, height: int) -> tuple:
    """Расширяет границы bbox на заданный padding с учетом границ изображения."""
    x1, y1, x2, y2 = map(int, bbox)
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def rasterize_bbox(width: int, height: int, bbox: tuple) -> np.ndarray:
    """Создает бинарную маску по координатам bbox."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox

    if y1 < y2 and x1 < x2:
        mask[y1:y2, x1:x2] = 255
    return mask


def apply_gaussian_blur(mask: np.ndarray, sigma: int) -> np.ndarray:
    """Применяет разделимое гауссово размытие."""
    radius = max(1, int(np.ceil(3 * sigma)))

    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(
        mask.astype(np.float32), ((radius, radius), (radius, radius)), mode="edge"
    )

    w_view = sliding_window_view(padded, window_shape=(2 * radius + 1), axis=1)
    blurred_w = np.tensordot(w_view, kernel, axes=([2], [0]))

    h_view = sliding_window_view(blurred_w, window_shape=(2 * radius + 1), axis=0)
    blurred = np.tensordot(h_view, kernel, axes=([2], [0]))

    return np.clip(blurred, 0, 255).astype(np.uint8)


def create_mask(
    width: int, height: int, bbox: list, blur: int, padding: int
) -> np.ndarray:  # На самом же деле на данном этапе не надо размывать маску, маска должна размываться в момент наложения маскировки на исходный кадр.
    """Расширяет, растеризует и размывает маску."""
    if not bbox:
        raise ValueError("bbox cannot be None")
    effective_bbox = (
        expand_bbox(bbox, padding, width, height)
        if padding > 0
        else tuple(map(int, bbox))
    )
    mask = rasterize_bbox(width, height, effective_bbox)
    """
    if blur is not None and float(blur) > 0:
        mask = apply_gaussian_blur(mask, blur)
    """
    return np.array(mask).astype(np.uint8)

def get_crop_params(bbox, padding, width, height):
    """Вычисляет параметры кропа без создания полной маски, сохраняя кратность 8."""
    x1, y1, x2, y2 = expand_bbox(bbox, padding, width, height)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    region_w = x2 - x1
    region_h = y2 - y1
    
    # Размеры кратные 8
    crop_w = int(np.ceil(region_w / 8.0) * 8)
    crop_h = int(np.ceil(region_h / 8.0) * 8)
    
    # Координаты кропа (центрируем относительно маски)
    left = cx - crop_w // 2
    top = cy - crop_h // 2
    
    # Сдвигаем кроп, если он выходит за границы изображения
    if left + crop_w > width:
        left = width - crop_w
    if left < 0:
        left = 0
        crop_w = (width // 8) * 8
        
    if top + crop_h > height:
        top = height - crop_h
    if top < 0:
        top = 0
        crop_h = (height // 8) * 8
        
    return (left, top, crop_w, crop_h), (x1, y1, x2, y2)


def create_cropped_mask(crop_w, crop_h, effective_bbox, left, top):
    """Создает маску только для кропа, без полной маски."""
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    
    # Переносим координаты bbox в систему координат кропа
    x1_c = max(0, effective_bbox[0] - left)
    y1_c = max(0, effective_bbox[1] - top)
    x2_c = min(crop_w, effective_bbox[2] - left)
    y2_c = min(crop_h, effective_bbox[3] - top)
    
    if y1_c < y2_c and x1_c < x2_c:
        mask[y1_c:y2_c, x1_c:x2_c] = 255
    return mask


def gaussian_blur_2d(tensor, sigma):
    """Гауссово размытие на GPU."""
    import torch.nn.functional as F
    kernel_size = int(2 * round(3 * sigma) + 1)
    x = torch.arange(kernel_size, device=tensor.device) - kernel_size // 2
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss /= gauss.sum()
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel.expand(tensor.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(tensor, kernel, padding=kernel_size//2, groups=tensor.shape[1])


def process_video_frame(frame_gpu, bbox, padding, blur_sigma, inpainting_fn):
    """
    Обработка одного кадра видеопотока с минимальным использованием памяти.
    frame_gpu: torch.Tensor (C, H, W) на GPU.
    """
    _, H, W = frame_gpu.shape
    
    # 1. Вычисляем параметры кропа (без создания маски)
    (left, top, crop_w, crop_h), eff_bbox = get_crop_params(bbox, padding, W, H)
    
    # 2. Создаем ТОЛЬКО кроп маски (маленький массив) и переносим на GPU
    crop_mask_np = create_cropped_mask(crop_w, crop_h, eff_bbox, left, top)
    crop_mask = torch.from_numpy(crop_mask_np).to(frame_gpu.device, dtype=torch.float32) / 255.0
    
    # 3. Вырезаем кроп изображения (view, без копирования)
    crop_img = frame_gpu[:, top:top+crop_h, left:left+crop_w]
    
    # 4. Нормализуем кроп изображения в диапазон [-1, 1]
    crop_img_normalized = crop_img.float() * 2.0 / 255.0 - 1.0
    
    # 5. Подготавливаем 4-канальный вход (N, 4, H, W)
    mask_channel = crop_mask.unsqueeze(0)
    x = torch.cat([mask_channel - 0.5, crop_img_normalized * mask_channel], dim=0).unsqueeze(0)
    
    # 6. Инпейнтинг
    with torch.no_grad():
        processed_crop = inpainting_fn(x)  # Ожидает (1, 4, H, W), возвращает (1, 3, H, W)
    processed_crop = processed_crop.squeeze(0)  # (3, Hc, Wc)
    
    # 7. Денормализация в [0, 255]
    processed_crop = (processed_crop * 0.5 + 0.5).clamp(0, 1) * 255.0
    
    # 8. Размытие маски на GPU (если нужно)
    if blur_sigma and blur_sigma > 0:
        blur_input = mask_channel.unsqueeze(0)
        blurred_mask = gaussian_blur_2d(blur_input, blur_sigma).squeeze(0)
        alpha = blurred_mask
    else:
        alpha = mask_channel
    
    # 9. Альфа-смешивание и обратная вставка
    result = frame_gpu.clone()
    roi = result[:, top:top+crop_h, left:left+crop_w].float()
    blended = (1.0 - alpha) * roi + alpha * processed_crop
    result[:, top:top+crop_h, left:left+crop_w] = blended.to(frame_gpu.dtype)
    
    return result


def process_video_frames_batch(frames_gpu, bboxes, padding, blur_sigma, inpainting_fn):
    """
    Пакетная обработка кадров видеопотока.
    frames_gpu: список из B тензоров формы (C, H, W) на GPU.
    bboxes: список из B bboxes (по одному на кадр) или один bbox для всех кадров.
    """
    B = len(frames_gpu)
    if B == 0:
        return []
        
    _, H, W = frames_gpu[0].shape
    device = frames_gpu[0].device
    
    # Нормализуем bboxes в список длиной B
    if not isinstance(bboxes[0], (list, tuple, np.ndarray)) or len(bboxes) != B:
        bboxes = [bboxes] * B
        
    # Чтобы объединить в батч, кропы должны быть одинакового размера.
    # Вычисляем параметры кропа для каждого кадра
    crop_params = [get_crop_params(bboxes[i], padding, W, H) for i in range(B)]
    
    # Определим максимальные crop_w и crop_h в батче, чтобы все кропы привести к одной форме
    max_crop_w = max(p[0][2] for p in crop_params)
    max_crop_h = max(p[0][3] for p in crop_params)
    
    adjusted_params = []
    for i in range(B):
        (left, top, crop_w, crop_h), eff_bbox = crop_params[i]
        # Центрируем кроп максимального размера относительно исходного центра
        cx = left + crop_w // 2
        cy = top + crop_h // 2
        
        n_left = cx - max_crop_w // 2
        n_top = cy - max_crop_h // 2
        
        # Корректируем границы
        if n_left + max_crop_w > W:
            n_left = W - max_crop_w
        if n_left < 0:
            n_left = 0
        if n_top + max_crop_h > H:
            n_top = H - max_crop_h
        if n_top < 0:
            n_top = 0
            
        adjusted_params.append(((n_left, n_top, max_crop_w, max_crop_h), eff_bbox))

    crop_imgs = []
    crop_masks = []
    
    for i in range(B):
        (left, top, crop_w, crop_h), eff_bbox = adjusted_params[i]
        
        # 1. Создаем кроп маски
        mask_np = create_cropped_mask(crop_w, crop_h, eff_bbox, left, top)
        mask_tensor = torch.from_numpy(mask_np).to(device, dtype=torch.float32) / 255.0
        crop_masks.append(mask_tensor.unsqueeze(0)) # (1, Hc, Wc)
        
        # 2. Вырезаем кроп изображения
        crop_img = frames_gpu[i][:, top:top+crop_h, left:left+crop_w]
        crop_img_normalized = crop_img.float() * 2.0 / 255.0 - 1.0
        crop_imgs.append(crop_img_normalized)
        
    # Объединяем в батч
    batch_imgs = torch.stack(crop_imgs, dim=0)
    batch_masks = torch.stack(crop_masks, dim=0)
    
    # 5. Подготавливаем 4-канальный вход (B, 4, Hc, Wc)
    x = torch.cat([batch_masks - 0.5, batch_imgs * batch_masks], dim=1)
    
    # 6. Инпейнтинг
    with torch.no_grad():
        processed_crops = inpainting_fn(x)  # Ожидает (B, 4, Hc, Wc), возвращает (B, 3, Hc, Wc)
        
    # 7. Денормализация в [0, 255]
    processed_crops = (processed_crops * 0.5 + 0.5).clamp(0, 1) * 255.0
    
    # 8. Размытие маски на GPU
    if blur_sigma and blur_sigma > 0:
        alpha = gaussian_blur_2d(batch_masks, blur_sigma)
    else:
        alpha = batch_masks
        
    # 9. Альфа-смешивание и вставка обратно для каждого кадра
    results = []
    for i in range(B):
        (left, top, crop_w, crop_h), _ = adjusted_params[i]
        res_frame = frames_gpu[i].clone()
        roi = res_frame[:, top:top+crop_h, left:left+crop_w].float()
        
        blended = (1.0 - alpha[i]) * roi + alpha[i] * processed_crops[i]
        res_frame[:, top:top+crop_h, left:left+crop_w] = blended.to(frames_gpu[i].dtype)
        results.append(res_frame)
        
    return results

def get_video_metadata(path_to_video: str) -> dict:
    with av.open(path_to_video) as container:
        video_stream = container.streams.video[0]
        metadata = {
            "width": video_stream.codec_context.width,
            "height": video_stream.codec_context.height,
            "fps": video_stream.average_rate,
            "time_base": video_stream.time_base,
            "pix_fmt": video_stream.codec_context.pix_fmt,
            "codec": video_stream.codec_context.name,
            "has_audio": len(container.streams.audio) > 0,
        }
        return metadata


def get_yolo_model(path_to_yolo_model: str, device: str | DeviceLikeType) -> YOLO:
    if not os.path.isfile(path_to_yolo_model):
        raise FileNotFoundError(f"Model file not found: {path_to_yolo_model}")
    device = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    logger.info(f"Loading YOLO model from {path_to_yolo_model} on device {device}")
    try:
        model = YOLO(path_to_yolo_model).to(device)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

def get_migan_model(path_to_migan_model: str, device: str | DeviceLikeType) -> torch.nn.Module:
    if not os.path.isfile(path_to_migan_model):
        raise FileNotFoundError(f"Model file not found: {path_to_migan_model}")
    device = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    
    state_dict = torch.load(path_to_migan_model, map_location=device)
    
    # Автоматическое определение разрешения из ключей весов
    resolution = 512 if any("b512" in k for k in state_dict.keys()) else 256
    logger.info(f"Loading PyTorch MI-GAN model ({resolution}x{resolution}) from {path_to_migan_model} on device {device}")
    
    try:
        model = MIGAN(resolution=resolution)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


def detect_scenes(video_path: str):
    """Определяет границы сцен в видео с помощью scenedetect."""
    try:
        from scenedetect import ContentDetector, detect
        logger.info(f"Начало поиска сцен в видео: {video_path}")
        scene_list = detect(video_path, ContentDetector())
        logger.info(f"Найдено сцен: {len(scene_list)}")
        return scene_list
    except Exception as e:
        logger.warning(f"Не удалось запустить scenedetect ({e}). Видео будет обработано как одна сцена.")
        return []


def get_union_bbox(bboxes):
    """Объединяет несколько ограничивающих прямоугольников в один общий."""
    if not bboxes:
        return None
    if not isinstance(bboxes[0], list):
        return bboxes
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def run_yolo_detection(yolo_model, frame_np):
    """Запускает YOLO для детекции вотермарок на кадре."""
    results = yolo_model(frame_np, verbose=False)
    bboxes = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            bboxes.append(coords)
    return bboxes


def main():
    args = create_parser().parse_args()
    
    yolo_model = None
    if args.path_to_yolo_model:
        yolo_model = get_yolo_model(args.path_to_yolo_model, args.device)
    elif not args.bbox:
        raise ValueError("Необходимо указать либо --bbox для ручного ввода, либо --path_to_yolo_model для автоматического поиска.")

    migan_model = get_migan_model(args.path_to_migan_model, args.device)
    
    # Загружаем метаданные видео
    video_metadata = get_video_metadata(args.input)
    logger.info(f"Метаданные видео: {video_metadata}")
    
    # Поиск сцен, если выбран режим static и нет ручного bbox
    scene_boundaries = []
    if args.mode == "static" and not args.bbox:
        scenes = detect_scenes(args.input)
        # Сохраняем номера кадров, на которых заканчиваются сцены
        scene_boundaries = [s[1].get_frames() for s in scenes]
        
    input_container = av.open(args.input)
    video_stream = input_container.streams.video[0]
    
    audio_stream = None
    if len(input_container.streams.audio) > 0:
        audio_stream = input_container.streams.audio[0]
        
    output_path = args.output
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, os.path.basename(args.input))
    else:
        _, ext = os.path.splitext(output_path)
        if not ext:
            _, input_ext = os.path.splitext(args.input)
            output_path += input_ext
            
    output_container = av.open(output_path, mode="w")
    
    # Создаем видеопоток на запись с кодеком h264
    output_video_stream = output_container.add_stream("libx264", rate=video_stream.average_rate)
    output_video_stream.width = video_stream.codec_context.width
    output_video_stream.height = video_stream.codec_context.height
    output_video_stream.pix_fmt = "yuv420p"
    output_video_stream.options = {"crf": "18"} # Высокое качество сжатия
    
    output_audio_stream = None
    if audio_stream:
        output_audio_stream = output_container.add_stream(audio_stream.codec_context.name, rate=audio_stream.rate)
        
    frame_buffer = []
    pts_buffer = []
    time_base_buffer = []
    
    # Переменная для хранения текущей статической маски/bbox сцены
    current_static_bbox = None
    frame_idx = 0
    
    def flush_buffer():
        nonlocal current_static_bbox
        if not frame_buffer:
            return
            
        bboxes_to_use = []
        if args.bbox:
            # Ручной bbox статичен для всего видео
            bboxes_to_use = [args.bbox] * len(frame_buffer)
        else:
            if args.mode == "static":
                # В статичном режиме ищем вотермарку один раз (на первом кадре батча/сцены)
                if current_static_bbox is None:
                    first_frame_np = frame_buffer[0].to_ndarray(format="rgb24")
                    detected_bboxes = run_yolo_detection(yolo_model, first_frame_np)
                    current_static_bbox = get_union_bbox(detected_bboxes)
                    if current_static_bbox is None:
                        current_static_bbox = [0, 0, 0, 0] # Пустой bbox
                bboxes_to_use = [current_static_bbox] * len(frame_buffer)
            else:
                # В динамичном режиме ищем вотермарку на каждом кадре
                for f in frame_buffer:
                    f_np = f.to_ndarray(format="rgb24")
                    detected_bboxes = run_yolo_detection(yolo_model, f_np)
                    union_bbox = get_union_bbox(detected_bboxes)
                    if union_bbox is None:
                        union_bbox = [0, 0, 0, 0]
                    bboxes_to_use.append(union_bbox)
                    
        # Подготавливаем тензоры на GPU
        device = torch.device(args.device)
        gpu_frames = []
        for f in frame_buffer:
            f_np = f.to_ndarray(format="rgb24")
            f_tensor = torch.from_numpy(f_np).permute(2, 0, 1).to(device)
            gpu_frames.append(f_tensor)
            
        # Запускаем пакетный инпейнринг
        processed_gpu_frames = process_video_frames_batch(
            gpu_frames, bboxes_to_use, args.padding, args.blur, migan_model
        )
        
        # Кодируем и сохраняем обработанные кадры
        for i, processed_tensor in enumerate(processed_gpu_frames):
            processed_np = processed_tensor.permute(1, 2, 0).byte().cpu().numpy()
            out_frame = av.VideoFrame.from_ndarray(processed_np, format="rgb24")
            out_frame.pts = pts_buffer[i]
            out_frame.time_base = time_base_buffer[i]
            
            for packet in output_video_stream.encode(out_frame):
                output_container.mux(packet)
                
        frame_buffer.clear()
        pts_buffer.clear()е
        time_base_buffer.clear()

    logger.info("Начало обработки видеопотока...")
    
    # Основной цикл демультиплексирования
    for packet in input_container.demux():
        if packet.stream.type == "video":
            for frame in packet.decode():
                # Если наступила граница новой сцены, сбрасываем кэш статического bbox
                if args.mode == "static" and frame_idx in scene_boundaries:
                    flush_buffer()
                    current_static_bbox = None
                    
                frame_buffer.append(frame)
                pts_buffer.append(frame.pts)
                time_base_buffer.append(frame.time_base)
                frame_idx += 1
                
                # Если бафер заполнен, обрабатываем батч
                if len(frame_buffer) >= args.batch_size:
                    flush_buffer()
                    
        elif packet.stream.type == "audio" and output_audio_stream:
            # Просто копируем аудиопакеты без декодирования
            packet.stream = output_audio_stream
            output_container.mux(packet)
            
    # Обрабатываем оставшиеся в буфере кадры
    flush_buffer()
    
    # Сбрасываем видеокодер
    for packet in output_video_stream.encode():
        output_container.mux(packet)
        
    input_container.close()
    output_container.close()
    logger.info(f"Обработка завершена успешно. Видео сохранено в: {output_path}")


if __name__ == "__main__":
    main()
