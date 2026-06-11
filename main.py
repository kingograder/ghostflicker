import argparse
import logging
import os

import av
import numpy as np
import torch
from numpy._core.numeric import require
from numpy.lib.stride_tricks import sliding_window_view
from torch._prims_common import DeviceLikeType
from ultralytics import YOLO

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

    # Настройки scenedetect
    # Настройки yolo
    parser.add_argument(
        "--path_to_yolo_model",
        required=False,
        type=str,
    )
    # Настройки video
    # Настройки migan
    parser.add_argument(
        "--path_to_migan_model",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--migan-quality",
        required=False,
        type=str,
        choices=["auto", "low", "medium"],
        default="auto",
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
        default=5,
        require=False,
        help="Внешний отступ для выделенной области.",  # Внешний отступ делать пропорциональным или строгим (проценты или пиксели?)
    )

    # Параметры для предварительного просмотра работы программы
    parser.add_argument(
        "--mask-preview",
        default=False,
        required=False,
        type="bool",
        help="При включении сохраняет маску для первого кадра видео в выходную директорию.",
    )

    parser.add_argument(
        "--inpaint-preview",
        default=False,
        required=False,
        type="bool",
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


def apply_gaussian_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
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
        mask = apply_gaussian_blur(mask, float(blur))
    """
    return np.array(mask).astype(np.uint8)

def crop_frame(image: np.ndarray, mask: np.ndarray, bbox: list, padding: int) -> tuple[np.ndarray, np.ndarray]:
    # Сначало найти центр замаскированной (белой) области
    # Потом обпределить размер замаскированной (белой) области
    # Сделать кроп до ближайшего размера кратного 8 так чтобы маска была внутри кропа целиком по-центру
    return cropped_image, cropped_mask

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
    logger.info(f"Using device: {device}")
    try:
        logger.info(f"Loading model from {path_to_yolo_model}")
        model = YOLO(path_to_yolo_model).to(device)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

def get_migan_model(path_to_migan_model: str, device: str) ->

def main():
    args = create_parser().parse_args()
    model = get_yolo_model(args.path_to_yolo_model, args.device)

    video_metadata = get_video_metadata(args.input)


if __name__ == "__main__":
    main()
