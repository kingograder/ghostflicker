"""CLI entry point for ghostflicker – watermark removal with YOLO + MI-GAN."""

import argparse
import logging
import os

import torch
from torch._prims_common import DeviceLikeType

from detection import SmartTracker, SceneDetector, SmartDetector, detect_bboxes
from pipeline import process_video, debug_yolo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Remove watermarks from video using YOLO detection and MI-GAN inpainting"
    )

    # I / O
    parser.add_argument("-i", "--input", required=True, help="Input video file path")
    parser.add_argument("-o", "--output", required=True, help="Output video file path")

    # Models
    parser.add_argument("--path_to_yolo_model", required=True, help="Path to YOLO .pt file")
    parser.add_argument(
        "--path_to_migan_model", required=True, help="Path to MI-GAN .pt file"
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda / cpu)",
    )

    # Detection mode
    parser.add_argument(
        "--detect_mode",
        choices=["per_frame", "scene", "smart"],
        default="per_frame",
        help="Detection mode (default: per_frame). "
        "--bbox flag overrides all modes with a static box.",
    )
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Static bbox – skip YOLO, use this box for every frame",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="YOLO confidence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.4,
        help="YOLO NMS IoU threshold (default: 0.4)",
    )
    parser.add_argument(
        "--scene_threshold",
        type=float,
        default=15.0,
        help="pyscenedetect content-detector sensitivity (lower = more scenes, default: 15.0)",
    )
    parser.add_argument(
        "--max_skip",
        type=int,
        default=100,
        help="Maximum Fibonacci frame skip in 'smart' mode (default: 100)",
    )

    # Mask
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Context padding (px) around the bbox before cropping (default: 20)",
    )
    parser.add_argument(
        "--mask_dilation",
        type=int,
        default=0,
        help="Expand the inpaint mask by N px on each side (default: 0)",
    )

    # Debug
    parser.add_argument(
        "--debug_yolo",
        action="store_true",
        help="Output a video with all YOLO detections drawn (no inpainting)",
    )

    return parser


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _validate_env(path: str, device: str | DeviceLikeType) -> None:
    """Check that *path* exists and that *device* is available."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is not available on this machine")


def get_yolo_model(path: str, device: str | DeviceLikeType):
    """Load an Ultralytics YOLO model onto *device*."""
    from ultralytics import YOLO

    _validate_env(path, device)
    device = torch.device(device)
    logger.info("Loading YOLO model from %s on device %s", path, device)
    try:
        model = YOLO(path).to(device)
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load YOLO model: %s", e, exc_info=True)
        raise


def get_migan_model(path: str, device: str | DeviceLikeType):
    """Load a MI-GAN Generator checkpoint, auto-detecting resolution (256/512)."""
    from migan_inference import Generator

    _validate_env(path, device)
    device = torch.device(device)
    logger.info("Loading MI-GAN model from %s on device %s", path, device)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Detect resolution from state-dict keys (b512 → 512, otherwise 256)
        resolution = 512 if any("b512" in k for k in state_dict) else 256
        logger.info("Detected MI-GAN resolution: %d", resolution)
        model = Generator(resolution=resolution)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        del state_dict
        logger.info("MI-GAN model loaded successfully")
        return model
    except Exception as e:
        logger.error("Failed to load MI-GAN model: %s", e, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, load models, build detection function, run pipeline."""
    args = create_parser().parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    yolo_model = get_yolo_model(args.path_to_yolo_model, device)
    migan_model = get_migan_model(args.path_to_migan_model, device) if not args.debug_yolo else None

    # Build the detection callable that produces bboxes per frame.
    if args.bbox:
        detection_fn = lambda _, __: [list(args.bbox)]
    elif args.detect_mode == "scene":
        logger.info("Scene detection mode: pre-computing scene boundaries …")
        detection_fn = SmartTracker(
            SceneDetector(args.input, yolo_model, args.conf, args.iou, args.scene_threshold)
        )
    elif args.detect_mode == "smart":
        logger.info(
            "Smart detection mode: Fibonacci frame skipping (max_skip=%d)",
            args.max_skip,
        )
        detection_fn = SmartDetector(
            args.input,
            yolo_model, args.conf, args.iou,
            max_skip=args.max_skip,
        )
    else:
        detection_fn = SmartTracker(
            lambda f, _: detect_bboxes(f, yolo_model, args.conf, args.iou)
        )

    # --debug_yolo shortcut: render detections, skip inpainting entirely.
    if args.debug_yolo:
        debug_yolo(args.input, args.output, detection_fn)
        return

    process_video(
        args.input,
        args.output,
        detection_fn,
        migan_model,
        device,
        args.padding,
        args.mask_dilation,
    )


if __name__ == "__main__":
    main()
