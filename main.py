import argparse
import logging
import os
from typing import Callable, List, Tuple, cast

import av
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scenedetect import AdaptiveDetector, SceneManager, detect, open_video
from torch._prims_common import DeviceLikeType
from tqdm import tqdm
from ultralytics import YOLO

from migan_inference import Generator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for watermark removal from video")

    # IO
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Input video file path"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Output video file path"
    )

    # Processing
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic"],
        default="dynamic",
        help="Mask processing mode",
    )
    parser.add_argument(
        "--skip_empty_scenes",
        action="store_true",
        help="Skip scenes without watermark",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    # Models & Device
    parser.add_argument("--path_to_yolo_model", required=True, type=str)
    parser.add_argument("--path_to_migan_model", required=True, type=str)
    parser.add_argument(
        "-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Hardware Acceleration
    parser.add_argument(
        "--use_gpu_decode",
        action="store_true",
        help="Use h264_cuvid for decoding",
    )
    parser.add_argument(
        "--use_gpu_encode",
        action="store_true",
        help="Use h264_nvenc for encoding",
    )

    # Mask settings
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("x1", "y1", "x2", "y2"))
    parser.add_argument("--blur", type=int, default=0)
    parser.add_argument(
        "--padding", type=int, default=20, help="Context padding around bbox for crop"
    )
    parser.add_argument(
        "--mask_dilation",
        type=int,
        default=0,
        help="Expand mask by N pixels on all sides",
    )

    # Previews
    parser.add_argument("--mask_preview", action="store_true")
    parser.add_argument("--inpaint_preview", action="store_true")

    return parser


def detect_scenes(
    path_to_video: str,
    threshold: float = 60.0,
    min_scene_len_frames: int = 15,
    return_frame_numbers: bool = True,
) -> List[int] | List[Tuple[float, float]]:
    """
    Detects scenes in video using PyAV backend.

    Args:
        video_path: Path to video file
        threshold: Sensitivity threshold (default 27.0)
        use_adaptive: Use AdaptiveDetector (resistant to camera motion)
        min_scene_len: Minimum scene length in frames (noise filter)
        return_frame_numbers: True -> list of first frame numbers of scenes.
                              False -> list of (start_seconds, end_seconds) tuples.
    """
    video = open_video(path_to_video, backend="pyav")
    scene_manager = SceneManager()
    detector = AdaptiveDetector(
        adaptive_threshold=threshold, min_scene_len=min_scene_len_frames
    )
    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video=video)
    scenes = scene_manager.get_scene_list()

    if not scenes:
        return []

    if return_frame_numbers:
        return [scene[0].frame_num for scene in scenes]
    else:
        return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scenes]


def apply_gaussian_blur(mask: np.ndarray, sigma: int) -> np.ndarray:
    """Applies separable Gaussian blur."""
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


def process_frame(
    frame: np.ndarray,
    bbox: list | None,
    yolo_model: YOLO,
    migan_model,
    device: torch.device,
    padding: int = 20,
    mask_dilation: int = 0,
) -> np.ndarray:
    import torch.nn.functional as F

    img_h, img_w = frame.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
    else:
        results = yolo_model(frame, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return frame
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    expanded = expand_bbox((x1, y1, x2, y2), padding, (img_h, img_w))
    cropped, adjusted = crop_frame_aligned(frame, expanded)

    crop_x1, crop_y1 = adjusted[0], adjusted[1]

    mask = np.ones(cropped.shape[:2], dtype=np.float32)
    mx1 = max(0, x1 - crop_x1 - mask_dilation)
    my1 = max(0, y1 - crop_y1 - mask_dilation)
    mx2 = min(cropped.shape[1], x2 - crop_x1 + mask_dilation)
    my2 = min(cropped.shape[0], y2 - crop_y1 + mask_dilation)
    mask[my1:my2, mx1:mx2] = 0.0

    model_res = migan_model.synthesis.resolution
    crop_h, crop_w = cropped.shape[:2]

    img_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() * 2 / 255 - 1
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    img_resized = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(model_res, model_res),
        mode="bilinear",
        align_corners=False,
    )[0]
    mask_resized = F.interpolate(
        mask_tensor, size=(model_res, model_res), mode="nearest"
    )[0]

    inp = (
        torch.cat([mask_resized - 0.5, img_resized * mask_resized], dim=0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out = migan_model(inp)

    out = F.interpolate(
        out, size=(crop_h, crop_w), mode="bilinear", align_corners=False
    )
    out = (out[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy() * 255

    mask_3ch = mask[:, :, np.newaxis]
    composed = cropped.astype(np.float32) * mask_3ch + out * (1 - mask_3ch)
    result = composed.clip(0, 255).astype(np.uint8)

    output = frame.copy()
    output[adjusted[1] : adjusted[3], adjusted[0] : adjusted[2]] = result

    return output


def process_video(
    path_to_input_video: str,
    path_to_output_video: str,
    bbox: list,
    yolo_model: YOLO,
    migan_model,
    device: torch.device,
    padding: int = 20,
    mask_dilation: int = 0,
):
    with (
        av.open(path_to_input_video) as input_container,
        av.open(path_to_output_video, mode="w") as output_container,
    ):
        if not input_container.streams.video:
            raise ValueError("Input file have no video stream")

        input_stream = input_container.streams.video[0]
        total_frames = input_stream.frames if input_stream.frames > 0 else None

        output_stream = output_container.add_stream(
            "libx264", rate=input_stream.average_rate
        )
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        output_stream.pix_fmt = "yuv420p"
        output_stream.time_base = input_stream.time_base

        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
        for frame in input_container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            image_processed = process_frame(
                image, bbox, yolo_model, migan_model, device, padding, mask_dilation
            )

            new_frame = av.VideoFrame.from_ndarray(image_processed, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            for packet in output_stream.encode(new_frame):
                output_container.mux(packet)

            pbar.update(1)
        pbar.close()

        for packet in output_stream.encode(None):
            output_container.mux(packet)


def get_video_metadata(path_to_video: str) -> dict:
    with av.open(path_to_video) as container:
        vs = container.streams.video[0]
        return {
            "width": vs.codec_context.width,
            "height": vs.codec_context.height,
            "fps": vs.average_rate,
            "time_base": vs.time_base,
            "pix_fmt": vs.codec_context.pix_fmt,
            "codec": vs.codec_context.name,
            "has_audio": len(container.streams.audio) > 0,
        }


def _validate_env(path: str, device: str | DeviceLikeType) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")


def get_yolo_model(path_to_yolo_model: str, device: str | DeviceLikeType) -> "YOLO":
    _validate_env(path_to_yolo_model, device)
    device = torch.device(device)
    logger.info(f"Loading YOLO model from {path_to_yolo_model} on device {device}")
    try:
        model = YOLO(path_to_yolo_model).to(device)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
        raise


def expand_bbox(bbox: list, padding: int, img_shape: tuple) -> tuple:
    """
    Expands bbox by padding pixels on all sides.
    Guarantees coordinates stay within image bounds.

    :param bbox: (x1, y1, x2, y2)
    :param padding: number of pixels to expand
    :param img_shape: (height, width) of the original frame
    :return: new bbox (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    h, w = img_shape

    x1_new = max(0, x1 - padding)
    y1_new = max(0, y1 - padding)
    x2_new = min(w, x2 + padding)
    y2_new = min(h, y2 + padding)

    return (x1_new, y1_new, x2_new, y2_new)


def crop_frame_aligned(frame: np.ndarray, bbox: tuple, min_size: int = 256) -> tuple:
    """
    Crops frame as a square with minimum size, centered on the bbox.

    :param frame: np.ndarray of the original frame
    :param bbox: (x1, y1, x2, y2) target region
    :param min_size: minimum crop size (default 256, legacy behavior)
    :return: (cropped_frame: np.ndarray, adjusted_bbox: tuple)
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = frame.shape[:2]

    max_req = max(x2 - x1, y2 - y1)

    if max_req <= 256:
        size = 256
    elif max_req <= 512:
        size = 512
    else:
        size = int(np.ceil(max_req / 64) * 64)

    size = max(size, min_size)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x1_new = cx - size // 2
    y1_new = cy - size // 2
    x2_new = x1_new + size
    y2_new = y1_new + size

    if x1_new < 0:
        x2_new -= x1_new
        x1_new = 0
    if y1_new < 0:
        y2_new -= y1_new
        y1_new = 0
    if x2_new > img_w:
        x1_new -= x2_new - img_w
        x2_new = img_w
    if y2_new > img_h:
        y1_new -= y2_new - img_h
        y2_new = img_h

    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(img_w, x2_new)
    y2_new = min(img_h, y2_new)

    adjusted_bbox = (x1_new, y1_new, x2_new, y2_new)
    cropped = frame[y1_new:y2_new, x1_new:x2_new].copy()

    return cropped, adjusted_bbox


def get_migan_model(path_to_migan_model: str, device: str | DeviceLikeType):
    _validate_env(path_to_migan_model, device)
    device = torch.device(device)
    logger.info(
        f"Loading PyTorch MI-GAN model from {path_to_migan_model} on device {device}"
    )
    try:
        state_dict = torch.load(
            path_to_migan_model, map_location=device, weights_only=True
        )
        resolution = 512 if any("b512" in k for k in state_dict) else 256
        logger.info(f"Detected MI-GAN resolution: {resolution}")
        model = Generator(resolution=resolution)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        del state_dict
        logger.info("MI-GAN model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load MI-GAN model: {e}", exc_info=True)
        raise


def main():
    args = create_parser().parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    yolo_model = get_yolo_model(args.path_to_yolo_model, device)
    migan_model = get_migan_model(args.path_to_migan_model, device)
    process_video(
        args.input,
        args.output,
        args.bbox,
        yolo_model,
        migan_model,
        device,
        args.padding,
        args.mask_dilation,
    )


if __name__ == "__main__":
    main()
