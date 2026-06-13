# GhostFlicker

> **WIP / Experimental** — this project is under active development and is not ready for production use. Expect breaking changes, incomplete features, and rough edges.

Automatic watermark removal from video using YOLO detection and MI-GAN inpainting.

## How it works

The pipeline runs in two decoupled stages:

1. **Stage 1 — Detection.** YOLO scans every frame (or a smart subset) to locate watermarks. Detection results are normalised, filtered for outliers, and stored as a per-frame bounding-box mapping. Optionally saved to a JSON file for reuse.

2. **Stage 2 — Inpainting.** Each flagged region is cropped, masked, and fed through MI-GAN to reconstruct the underlying content. The inpainted patch is composited back into the original frame.

This two-stage design ensures detection accuracy, eliminates outliers and artefacts, and allows re-running inpainting without re-detecting.

## Installation

Requires Python 3.9+ and [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Clone the repository
git clone https://github.com/<your-username>/ghostflicker.git
cd ghostflicker

# Install dependencies
uv sync          # or: pip install -e .
```

### Models

Place your YOLO and MI-GAN model files under `models/`:

```
models/
  yolo/
    best.pt                    # your trained YOLO watermark detector
  migan/
    migan_512_places2.pt       # MI-GAN inpainting weights (256 or 512)
```

## Usage

### Full pipeline (detect + inpaint)

```bash
python main.py \
  -i input.mp4 \
  -o output.mp4 \
  --path_to_yolo_model models/yolo/best.pt \
  --path_to_migan_model models/migan/migan_512_places2.pt \
  --detect_mode smart
```

### Save detections to JSON (skip inpainting)

```bash
python main.py \
  -i input.mp4 \
  -o /dev/null \
  --path_to_yolo_model models/yolo/best.pt \
  --detect_mode smart \
  --save_detection_file detections.json \
  --debug_yolo
```

### Re-run inpainting from saved detections

```bash
python main.py \
  -i input.mp4 \
  -o output.mp4 \
  --path_to_yolo_model models/yolo/best.pt \
  --path_to_migan_model models/migan/migan_512_places2.pt \
  --detection_file detections.json
```

### Debug: draw detections without inpainting

```bash
python main.py \
  -i input.mp4 \
  -o debug.mp4 \
  --path_to_yolo_model models/yolo/best.pt \
  --debug_yolo
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *required* | Input video file |
| `-o`, `--output` | *required* | Output video file |
| `--path_to_yolo_model` | *required* | Path to YOLO `.pt` file |
| `--path_to_migan_model` | — | Path to MI-GAN `.pt` file (required unless `--debug_yolo`) |
| `-d`, `--device` | `cuda` / `cpu` | Inference device |
| `--detect_mode` | `per_frame` | `per_frame`, `scene`, or `smart` |
| `--bbox` | — | Static `[x1 y1 x2 y2]` — skip YOLO, use this box every frame |
| `--conf` | `0.05` | YOLO confidence threshold |
| `--iou` | `0.4` | YOLO NMS IoU threshold |
| `--scene_threshold` | `15.0` | Scene-detection sensitivity (scene mode only) |
| `--max_skip` | `100` | Max Fibonacci skip (smart mode only) |
| `--padding` | `20` | Context padding around bbox before cropping (px) |
| `--mask_dilation` | `0` | Expand inpaint mask by N px on each side |
| `--detection_file` | — | Load detections from JSON, skip Stage 1 |
| `--save_detection_file` | — | Save detections to JSON after Stage 1 |
| `--debug_yolo` | — | Draw detections only, skip inpainting |

### Detection modes

- **`per_frame`** — YOLO runs on every frame. Slowest but most accurate.
- **`scene`** — YOLO runs on the first frame of each scene (via pyscenedetect). Faster for videos with long static shots.
- **`smart`** — Fibonacci-skip: YOLO runs on key frames, binary-searches transitions. Fastest with good accuracy.
- **`--bbox`** — Bypasses YOLO entirely. Uses the given static box for every frame.

## Project structure

```
ghostflicker/
  main.py              # CLI entry point
  detection.py         # Stage 1: YOLO detection + normalisation
  pipeline.py          # Stage 2: cropping, masking, MI-GAN inpainting
  migan_inference.py   # MI-GAN generator architecture
  models/
    yolo/              # YOLO weights
    migan/             # MI-GAN weights
```

## TODO

- [ ] Batch processing
- [ ] GPU memory optimisation (currently loads full model per run)
- [ ] Adjustable output codec / quality settings
- [ ] Audio passthrough (currently stripped in output)
- [ ] CLI progress reporting improvements
