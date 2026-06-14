import argparse
import logging
import os

import torch

from detection import load_detections, run_detection_stage, save_detections
from pipeline import debug_yolo, process_video

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_parser():
    p = argparse.ArgumentParser(description="Watermark removal CLI.")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--path_to_yolo_model", required=True)
    p.add_argument("--path_to_migan_model")
    p.add_argument(
        "-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--detect_mode", choices=["per_frame", "scene", "smart"], default="per_frame"
    )
    p.add_argument("--bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.4)
    p.add_argument("--scene_threshold", type=float, default=15.0)
    p.add_argument("--max_skip", type=int, default=100)
    p.add_argument("--padding", type=int, default=20)
    p.add_argument("--mask_dilation", type=int, default=0)
    p.add_argument("--detection_file")
    p.add_argument("--save_detection_file")
    p.add_argument("--debug_yolo", action="store_true")
    return p


def _check_file(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def get_yolo_model(path, device):
    from ultralytics import YOLO

    _check_file(path, "YOLO")
    return YOLO(path).to(device)


def get_migan_model(path, device):
    from migan_inference import Generator

    _check_file(path, "MI-GAN")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    res = 512 if any("b512" in k for k in state_dict) else 256
    model = Generator(resolution=res)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    del state_dict
    return model


def main():
    args = create_parser().parse_args()

    # Fail fast: валидация окружения и входных данных
    _check_file(args.input, "Input")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    if not args.debug_yolo and not args.path_to_migan_model:
        raise ValueError("MI-GAN path required.")

    # Stage 1: Detection
    if args.detection_file:
        detections = load_detections(args.detection_file)
    else:
        yolo = get_yolo_model(args.path_to_yolo_model, device)
        detections = run_detection_stage(
            args.input,
            yolo,
            args.detect_mode,
            args.conf,
            args.iou,
            args.scene_threshold,
            args.max_skip,
            args.bbox,
        )
        if args.save_detection_file:
            save_detections(detections, args.save_detection_file, args.input)

    if args.debug_yolo:
        debug_yolo(args.input, args.output, detections)
        return

    # Stage 2: Inpainting
    migan = get_migan_model(args.path_to_migan_model, device)
    process_video(
        args.input,
        args.output,
        detections,
        migan,
        device,
        args.padding,
        args.mask_dilation,
    )


if __name__ == "__main__":
    main()
