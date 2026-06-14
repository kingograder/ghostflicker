import logging

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

logger = logging.getLogger(__name__)


def crop_frame_aligned(frame, bbox, min_size=256):
    # Square crop centered on bbox, edge-padded if near boundary
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    size = max(x2 - x1, y2 - y1)
    size = 256 if size <= 256 else 512 if size <= 512 else int(np.ceil(size / 64) * 64)
    size = max(size, min_size)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    vx1, vy1 = cx - size // 2, cy - size // 2
    vx2, vy2 = vx1 + size, vy1 + size

    pl, pt = max(0, -vx1), max(0, -vy1)
    pr, pb = max(0, vx2 - w), max(0, vy2 - h)

    cx1, cy1 = max(0, vx1), max(0, vy1)
    cx2, cy2 = min(w, vx2), min(h, vy2)
    valid = frame[cy1:cy2, cx1:cx2]

    cropped = (
        np.pad(valid, ((pt, pb), (pl, pr), (0, 0)), mode="edge")
        if (pl or pt or pr or pb)
        else valid.copy()
    )
    return cropped, (cx1, cy1, cx2, cy2), (pl, pt)


def process_frame(frame, bboxes, model, device, padding=20, mask_dilation=0):
    # Inpaint all watermark regions in frame using MI-GAN
    h, w = frame.shape[:2]
    res = model.synthesis.resolution

    for x1, y1, x2, y2 in bboxes:
        ex1, ey1 = max(0, x1 - padding), max(0, y1 - padding)
        ex2, ey2 = min(w, x2 + padding), min(h, y2 + padding)
        cropped, (cx1, cy1, cx2, cy2), (pl, pt) = crop_frame_aligned(
            frame, (ex1, ey1, ex2, ey2)
        )
        ch, cw = cropped.shape[:2]

        mask = np.ones((ch, cw), dtype=np.float32)
        mx1 = max(0, x1 - cx1 + pl - mask_dilation)
        my1 = max(0, y1 - cy1 + pt - mask_dilation)
        mx2 = min(cw, x2 - cx1 + pl + mask_dilation)
        my2 = min(ch, y2 - cy1 + pt + mask_dilation)
        mask[my1:my2, mx1:mx2] = 0.0

        img_t = torch.from_numpy(cropped).permute(2, 0, 1).float().div(127.5).sub(1.0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        img_r = F.interpolate(
            img_t.unsqueeze(0), size=res, mode="bicubic", align_corners=False
        )[0].clamp(-1, 1)
        mask_r = F.interpolate(mask_t.unsqueeze(0), size=res, mode="nearest")[0]

        inp = torch.cat([mask_r - 0.5, img_r * mask_r], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)

        out = F.interpolate(out, size=(ch, cw), mode="bicubic", align_corners=False)[0]
        out = (
            out.cpu()
            .permute(1, 2, 0)
            .mul(127.5)
            .add(127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )

        # Composite and paste back
        hole = mask == 0.0
        cropped[hole] = out[hole]

        vh, vw = cy2 - cy1, cx2 - cx1
        frame[cy1:cy2, cx1:cx2] = cropped[pt : pt + vh, pl : pl + vw]

    return frame


def process_video(
    in_path, out_path, detections, model, device, padding=20, mask_dilation=0
):
    # Stage 2: inpaint pre-computed watermark detections
    with av.open(in_path) as inp, av.open(out_path, "w") as out:
        if not inp.streams.video:
            raise ValueError("No video stream")
        vs = inp.streams.video[0]
        os_ = out.add_stream("libx264", rate=vs.average_rate)
        os_.width, os_.height, os_.pix_fmt, os_.time_base = (
            vs.width,
            vs.height,
            "yuv420p",
            vs.time_base,
        )

        for i, frame in enumerate(
            tqdm(inp.decode(video=0), total=vs.frames or None, desc="Stage 2")
        ):
            img = frame.to_ndarray(format="rgb24")
            if bboxes := detections.get(i, []):
                img = process_frame(img, bboxes, model, device, padding, mask_dilation)

            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf.pts, vf.time_base = frame.pts, frame.time_base
            for p in os_.encode(vf):
                out.mux(p)

        for p in os_.encode(None):
            out.mux(p)


def debug_yolo(in_path, out_path, detections):
    # Draw pre-computed bboxes onto each frame for visual inspection
    with av.open(in_path) as inp, av.open(out_path, "w") as out:
        vs = inp.streams.video[0]
        os_ = out.add_stream("libx264", rate=vs.average_rate)
        os_.width, os_.height, os_.pix_fmt = vs.width, vs.height, "yuv420p"

        for i, frame in enumerate(
            tqdm(inp.decode(video=0), total=vs.frames or None, desc="Debug")
        ):
            img = frame.to_ndarray(format="rgb24")
            if bboxes := detections.get(i, []):
                pil = Image.fromarray(img)
                draw = ImageDraw.Draw(pil)
                for b in bboxes:
                    draw.rectangle(b, outline="green", width=2)
                img = np.array(pil)

            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf.pts, vf.time_base = frame.pts, frame.time_base
            for p in os_.encode(vf):
                out.mux(p)

        for p in os_.encode(None):
            out.mux(p)
