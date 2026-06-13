"""Core pipeline: frame/video processing, cropping, and MI-GAN inpainting.

Stage 2 of the watermark removal pipeline.  Takes pre-computed detection
results (produced by Stage 1 in ``detection.py``) and applies
MI-GAN inpainting to every flagged region.
"""

import logging

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crop utilities
# ---------------------------------------------------------------------------


def expand_bbox(bbox: list, padding: int, img_shape: tuple) -> tuple:
    """Expand bbox by ``padding`` pixels on each side, clamped to image."""
    x1, y1, x2, y2 = bbox
    h, w = img_shape
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(w, x2 + padding),
        min(h, y2 + padding),
    )


def pad_frame_edges(
    frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Extract a rectangular region, padding out-of-bounds areas with edge replication.

    Returns (padded_crop, (pad_left, pad_top)) where (pad_left, pad_top)
    indicate where the valid frame region starts inside the returned crop.
    """
    img_h, img_w = frame.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    x1_v = max(0, x1)
    y1_v = max(0, y1)
    x2_v = min(img_w, x2)
    y2_v = min(img_h, y2)

    valid = frame[y1_v:y2_v, x1_v:x2_v]

    if pad_left or pad_top or pad_right or pad_bottom:
        cropped = np.pad(
            valid,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
    else:
        cropped = valid.copy()

    return cropped, (pad_left, pad_top)


def crop_frame_aligned(
    frame: np.ndarray, bbox: tuple, min_size: int = 256
) -> tuple[np.ndarray, tuple, tuple[int, int]]:
    """
    Square crop centered on *bbox*, sized to the next standard resolution.

    Sizes: 256 → 512 → ceil(bbox_size / 64) * 64. Out-of-bounds regions are
    edge-padded via :func:`pad_frame_edges`.

    Returns (cropped_img, valid_bbox_in_frame_coords, (pad_left, pad_top)).
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = frame.shape[:2]

    size = max(x2 - x1, y2 - y1)
    if size <= 256:
        size = 256
    elif size <= 512:
        size = 512
    else:
        size = int(np.ceil(size / 64) * 64)
    size = max(size, min_size)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x1_virt = cx - size // 2
    y1_virt = cy - size // 2
    x2_virt = x1_virt + size
    y2_virt = y1_virt + size

    cropped, (pad_left, pad_top) = pad_frame_edges(frame, x1_virt, y1_virt, x2_virt, y2_virt)

    valid_bbox = (
        max(0, x1_virt),
        max(0, y1_virt),
        min(img_w, x2_virt),
        min(img_h, y2_virt),
    )

    return cropped, valid_bbox, (pad_left, pad_top)


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------


def get_video_metadata(path_to_video: str) -> dict:
    """Extract stream metadata from a video file via PyAV."""
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


# ---------------------------------------------------------------------------
# Core frame & video processing
# ---------------------------------------------------------------------------


def process_frame(
    frame: np.ndarray,
    bboxes: list[list[int]],
    migan_model,
    device: torch.device,
    padding: int = 20,
    mask_dilation: int = 0,
) -> np.ndarray:
    """
    Inpaint all watermark regions in *frame* using MI-GAN.

    For each (x1,y1,x2,y2) in *bboxes*:
        1. Expand with context padding.
        2. Square crop centered on the expanded area (edge-padded if needed).
        3. Binary mask: 0 = hole (watermark), 1 = known (background).
        4. Preprocess: resize to model resolution, build 4-channel MI-GAN input
           ``[mask - 0.5, image * mask]``.
        5. MI-GAN inference → output image inpainting.
        6. Resize back to crop size, composite original × mask + out × (1 - mask).
        7. Paste the valid (non-padded) region back into the frame.

    Multiple bboxes are processed sequentially on the same frame.

    Args:
        frame: RGB image (H, W, 3). Modified in place.
        bboxes: List of (x1, y1, x2, y2) bounding boxes.
        migan_model: MI-GAN Generator instance.
        device: Torch device for inference.
        padding: Context pixels added around the bbox before cropping.
        mask_dilation: Extra pixels added to the mask (expands the inpaint area).

    Returns:
        Processed RGB frame (same reference as input *frame*).
    """
    img_h, img_w = frame.shape[:2]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        # ---- 1. Expand bbox for context -----------------------------------------------------------------
        expanded = expand_bbox((x1, y1, x2, y2), padding, (img_h, img_w))

        # ---- 2. Square crop (edge-padded if near boundary) --------------------------------------------
        cropped, valid_bbox, (pad_left, pad_top) = crop_frame_aligned(frame, expanded)
        crop_x1, crop_y1 = valid_bbox[0], valid_bbox[1]

        # ---- 3. Binary mask (0 = hole / inpaint, 1 = known / keep) ----------------------------------
        mask = np.ones(cropped.shape[:2], dtype=np.float32)
        # Mask coordinates in the *padded* crop space: account for both the valid-region
        # offset (crop_x1) and the edge-padding offset (pad_left).
        mx1 = max(0, x1 - crop_x1 + pad_left - mask_dilation)
        my1 = max(0, y1 - crop_y1 + pad_top - mask_dilation)
        mx2 = min(cropped.shape[1], x2 - crop_x1 + pad_left + mask_dilation)
        my2 = min(cropped.shape[0], y2 - crop_y1 + pad_top + mask_dilation)
        mask[my1:my2, mx1:mx2] = 0.0

        model_res = migan_model.synthesis.resolution
        crop_h, crop_w = cropped.shape[:2]

        # ---- 4. Preprocess for MI-GAN -----------------------------------------------------------------
        # Normalise image to [-1, 1].  The mask stays as {0, 1}.
        img_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() * (2.0 / 255.0) - 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Resize both to the model's native resolution.
        # Image: bicubic (clamped to avoid overshoot), mask: nearest (preserves 0/1).
        img_resized = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(model_res, model_res),
            mode="bicubic",
            align_corners=False,
        )[0].clamp(-1.0, 1.0)
        mask_resized = F.interpolate(
            mask_tensor, size=(model_res, model_res), mode="nearest"
        )[0]

        # MI-GAN expects 4-channel input: [mask - 0.5, image * mask].
        #   mask - 0.5  →  known = +0.5, hole = -0.5
        #   image * mask → hole pixels zeroed out, known pixels preserved
        inp = (
            torch.cat([mask_resized - 0.5, img_resized * mask_resized], dim=0)
            .unsqueeze(0)
            .to(device)
        )

        # ---- 5. MI-GAN inference ---------------------------------------------------------------------
        with torch.no_grad():
            out = migan_model(inp)

        # ---- 6. Postprocess & composite --------------------------------------------------------------
        # Resize model output back to the crop resolution.
        out = F.interpolate(out, size=(crop_h, crop_w), mode="bicubic", align_corners=False)
        # Denormalise from [-1, 1] to [0, 255] uint8.
        out = (out[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0.0, 1.0).numpy()
        out = (out * 255.0).astype(np.uint8)

        # Composite: keep original where mask=1 (background), use inpainted where mask=0 (hole).
        mask_3ch = mask[:, :, np.newaxis]
        composed = (
            cropped.astype(np.float32) * mask_3ch
            + out.astype(np.float32) * (1.0 - mask_3ch)
        )
        result = composed.clip(0.0, 255.0).astype(np.uint8)

        # ---- 7. Paste valid region back into frame ----------------------------------------------------
        valid_h = valid_bbox[3] - valid_bbox[1]
        valid_w = valid_bbox[2] - valid_bbox[0]
        valid_region = result[pad_top : pad_top + valid_h, pad_left : pad_left + valid_w]
        frame[valid_bbox[1] : valid_bbox[3], valid_bbox[0] : valid_bbox[2]] = valid_region

    return frame


def process_video(
    path_to_input_video: str,
    path_to_output_video: str,
    detections: dict[int, list[list[int]]],
    migan_model,
    device: torch.device,
    padding: int = 20,
    mask_dilation: int = 0,
) -> None:
    """Stage 2: inpaint pre-computed watermark detections.

    Reads the input video frame by frame, looks up bboxes from
    *detections* for the current ``frame_idx``, and applies MI-GAN
    inpainting via :func:`process_frame`.

    Args:
        path_to_input_video: Source video file.
        path_to_output_video: Destination video file (created / overwritten).
        detections: Pre-computed mapping ``frame_idx -> list[bbox]``.
            Produced by :func:`detection.run_detection_stage`.
        migan_model: MI-GAN Generator instance.
        device: Torch device for inference.
        padding: Context padding (px) around bbox before cropping.
        mask_dilation: Extra pixels added to the inpaint mask on each side.
    """
    with (
        av.open(path_to_input_video) as input_container,
        av.open(path_to_output_video, mode="w") as output_container,
    ):
        if not input_container.streams.video:
            raise ValueError("Input file has no video stream")

        input_stream = input_container.streams.video[0]
        total_frames = input_stream.frames if input_stream.frames > 0 else None

        output_stream = output_container.add_stream(
            "libx264", rate=input_stream.average_rate
        )
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        output_stream.pix_fmt = "yuv420p"
        output_stream.time_base = input_stream.time_base

        pbar = tqdm(total=total_frames, desc="Stage 2: inpainting", unit="frame")
        for frame_idx, frame in enumerate(input_container.decode(video=0)):
            image = frame.to_ndarray(format="rgb24")

            # Look up pre-computed bboxes for this frame (empty list = skip).
            bboxes = detections.get(frame_idx, [])
            if bboxes:
                image = process_frame(image, bboxes, migan_model, device, padding, mask_dilation)

            new_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            for packet in output_stream.encode(new_frame):
                output_container.mux(packet)

            pbar.update(1)
        pbar.close()

        for packet in output_stream.encode(None):
            output_container.mux(packet)


def debug_yolo(
    path_to_input_video: str,
    path_to_output_video: str,
    detections: dict[int, list[list[int]]],
) -> None:
    """Draw pre-computed bboxes onto each frame for visual inspection.

    Reads the input video, looks up bboxes from *detections* for each
    frame, draws them as green rectangles, and writes the result to a
    video file.  No inpainting is performed — purely diagnostic.

    Args:
        path_to_input_video: Source video file.
        path_to_output_video: Destination video file (created / overwritten).
        detections: Pre-computed mapping ``frame_idx -> list[bbox]``.
    """
    input_container = av.open(path_to_input_video)
    in_stream = input_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    fps = in_stream.average_rate

    output_container = av.open(path_to_output_video, mode="w")
    out_stream = output_container.add_stream("libx264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"

    total_frames = in_stream.frames or 0
    pbar = tqdm(total=total_frames, desc="Drawing detections", unit="frame")

    for frame_idx, frame in enumerate(input_container.decode(video=0)):
        image = frame.to_ndarray(format="rgb24")

        bboxes = detections.get(frame_idx, [])
        if bboxes:
            pil_img = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_img)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            image = np.array(pil_img)

        new_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        for packet in out_stream.encode(new_frame):
            output_container.mux(packet)

        pbar.update(1)

    pbar.close()
    for packet in out_stream.encode(None):
        output_container.mux(packet)
    output_container.close()
    input_container.close()
    logger.info("Debug video written to %s", path_to_output_video)
