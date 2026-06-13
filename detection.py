"""YOLO-based watermark detection with smart tracking.

This module contains everything needed for Stage 1 of the pipeline:

- Raw YOLO detection (``detect_bboxes``).
- Normalisation wrappers (``SmartTracker``, ``SceneDetector``,
  ``SmartDetector``) that filter, merge and carry-forward bboxes.
- Full-video detection pass (``run_detection_stage``) that scans the
  entire video and returns a ``dict[int, list[bbox]]``.
- JSON serialisation (``save_detections`` / ``load_detections``).
"""

import json
import logging
import os
from typing import Callable

import numpy as np
from tqdm import tqdm

import av
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def detect_bboxes(
    frame: np.ndarray,
    yolo_model: YOLO,
    conf: float = 0.05,
    iou: float = 0.4,
) -> list[list[int]]:
    results = yolo_model(frame, verbose=False, conf=conf, iou=iou)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return []
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes]


def _merge_bboxes(detected: list[list[int]]) -> list[int]:
    if not detected:
        return None
    xs = [d[0] for d in detected]
    ys = [d[1] for d in detected]
    xe = [d[2] for d in detected]
    ye = [d[3] for d in detected]
    return [min(xs), min(ys), max(xe), max(ye)]


def _compute_iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(a[2] - a[0], 1) * max(a[3] - a[1], 1)
    area_b = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
    return inter / (area_a + area_b - inter)


def _boxes_overlap(a: list[int], b: list[int], gap: int = 100) -> bool:
    x1 = max(a[0] - gap, b[0] - gap)
    y1 = max(a[1] - gap, b[1] - gap)
    x2 = min(a[2] + gap, b[2] + gap)
    y2 = min(a[3] + gap, b[3] + gap)
    return x2 > x1 and y2 > y1


def _group_and_merge_bboxes(detected: list[list[int]], gap: int = 100) -> list[list[int]]:
    if not detected:
        return []
    n = len(detected)
    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(detected[i], detected[j], gap):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(detected[i])

    merged_results = []
    for g_boxes in groups.values():
        xs = [b[0] for b in g_boxes]
        ys = [b[1] for b in g_boxes]
        xe = [b[2] for b in g_boxes]
        ye = [b[3] for b in g_boxes]
        merged_results.append([min(xs), min(ys), max(xe), max(ye)])
    return merged_results


def _compute_multi_iou(list_a: list[list[int]], list_b: list[list[int]]) -> float:
    if not list_a or not list_b:
        return 0.0
    ious = []
    for a in list_a:
        max_iou = 0.0
        for b in list_b:
            max_iou = max(max_iou, _compute_iou(a, b))
        ious.append(max_iou)
    return sum(ious) / len(ious)


# ---------------------------------------------------------------------------
# Universal filter wrapper
# ---------------------------------------------------------------------------



class SmartTracker:
    """
    Universal filter for any detection function.

    - Merges multiple YOLO bboxes into one (handles split detection).
    - Rejects spatial outliers (sudden area change + low IoU).
    - Noise gate: tiny bbox jitter (IoU >= threshold) keeps the old box.
    - Carry-forward on empty detection from wrapped function.

    Args:
        detect_fn: Underlying detector ``(frame, frame_idx) -> list[bbox]``.
        max_misses: How many consecutive misses to carry forward.
        pixel_diff_threshold: Max mean pixel diff before stopping carry.
        iou_noise_threshold: IoU above this = jitter, keep old bbox.
        expansion_factor: Area ratio above this + low IoU = outlier reject.
    """

    def __init__(
        self,
        detect_fn: Callable[[np.ndarray, int], list[list[int]]],
        max_misses: int = 5,
        pixel_diff_threshold: float = 10.0,
        iou_noise_threshold: float = 0.95,
        expansion_factor: float = 3.0,
    ):
        self.detect_fn = detect_fn
        self.max_misses = max_misses
        self.pixel_diff_threshold = pixel_diff_threshold
        self.iou_noise_threshold = iou_noise_threshold
        self.expansion_factor = expansion_factor
        self._last_bbox: list[int] | None = None
        self._miss_count: int = 0
        self._prev_pixels: np.ndarray | None = None

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[list[int]]:
        detected = self.detect_fn(frame, frame_idx)

        if detected:
            merged = _merge_bboxes(detected)

            if self._last_bbox is not None:
                iou_val = _compute_iou(merged, self._last_bbox)
                a_new = (merged[2] - merged[0]) * (merged[3] - merged[1])
                a_old = (self._last_bbox[2] - self._last_bbox[0]) * (self._last_bbox[3] - self._last_bbox[1])
                area_ratio = max(a_new, a_old) / max(min(a_new, a_old), 1)

                if area_ratio > self.expansion_factor and iou_val < 0.3:
                    return [self._last_bbox]

                if iou_val >= self.iou_noise_threshold:
                    return [self._last_bbox]

            self._last_bbox = merged
            self._miss_count = 0
            x1, y1, x2, y2 = merged
            self._prev_pixels = frame[y1:y2, x1:x2].copy()
            return [merged]

        if self._last_bbox is None:
            return []

        self._miss_count += 1
        if self._miss_count > self.max_misses:
            return []

        x1, y1, x2, y2 = self._last_bbox
        cur = frame[y1:y2, x1:x2]
        if self._prev_pixels is not None and cur.shape == self._prev_pixels.shape:
            diff = float(np.mean(np.abs(cur.astype(float) - self._prev_pixels.astype(float))))
            if diff >= self.pixel_diff_threshold:
                return []

        self._prev_pixels = cur.copy()
        return [self._last_bbox]


# ---------------------------------------------------------------------------
# Scene detector — self-contained
# ---------------------------------------------------------------------------


def _detect_scenes(video_path: str, threshold: float) -> list[int]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError:
        raise ImportError(
            "Scene-detection mode requires `scenedetect`. "
            "Install it with: pip install scenedetect"
        )

    video = open_video(video_path, backend="pyav")
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)

    scene_list = manager.get_scene_list()
    if not scene_list:
        return []

    return sorted({start.get_frames() for start, _ in scene_list})


class SceneDetector:
    """
    Runs YOLO on the first frame of each scene; keeps that bbox for the
    entire scene.  Always returns a bbox on every frame (never []).

    Args:
        video_path: Path to video (for pyscenedetect pre-pass).
        yolo_model: Loaded YOLO model.
        conf, iou: YOLO parameters.
        scene_threshold: Content-detector sensitivity.
    """

    def __init__(self, video_path: str, yolo_model, conf: float, iou: float,
                 scene_threshold: float = 15.0):
        self.yolo_model = yolo_model
        self.conf = conf
        self.iou = iou
        self._boundaries = _detect_scenes(video_path, scene_threshold)
        self._boundary_idx = 0
        self._last_bbox: list[int] | None = None

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[list[int]]:
        is_boundary = (
            frame_idx == 0 or
            (self._boundary_idx < len(self._boundaries)
             and frame_idx >= self._boundaries[self._boundary_idx])
        )
        if is_boundary:
            if frame_idx != 0:
                self._boundary_idx += 1
            detected = detect_bboxes(frame, self.yolo_model, self.conf, self.iou)
            if detected:
                self._last_bbox = _merge_bboxes(detected)
        return [self._last_bbox] if self._last_bbox is not None else []


# ---------------------------------------------------------------------------
# Smart (Fibonacci-skip) detector — self-contained
# ---------------------------------------------------------------------------


class SmartDetector:
    """Fibonacci-skip detector with binary-search transition detection.

    YOLO runs on spaced-out frames (1, 2, 3, 5, 8, 13...).  Between checks
    the last known bbox is carried forward.  When YOLO confirms a position
    change (IoU drops below threshold), binary-searches the accumulated
    frame buffer to find the exact transition frame.  On YOLO miss, uses a
    short skip (2 frames) instead of the full Fibonacci skip.

    Args:
        video_path: Path to the input video.
        yolo_model: Loaded YOLO model.
        conf, iou: YOLO parameters.
        iou_threshold: IoU below this = bbox position changed.
        max_skip: Maximum Fibonacci skip before reset to 1.
    """

    def __init__(self, video_path: str, yolo_model, conf: float, iou: float,
                 iou_threshold: float = 0.7, max_skip: int = 100):
        self.yolo_model = yolo_model
        self.conf = conf
        self.iou = iou
        self.iou_threshold = iou_threshold
        self.max_skip = max_skip

        self._precalculated_bboxes: dict[int, list[list[int]]] = {}
        self._yolo_count: int = 0
        self._precalculate(video_path)

    def _precalculate(self, video_path: str):
        """Scan the entire video with YOLO + Fibonacci skipping.

        Decodes every frame, runs YOLO on key frames, binary-searches
        transitions, and builds a normalised per-frame bbox mapping.
        A progress bar is shown during the video scan.
        """
        import av

        container = av.open(video_path)
        if not container.streams.video:
            raise ValueError("Input file has no video stream")

        # Total frames for the progress bar.
        in_stream = container.streams.video[0]
        total_frames = in_stream.frames if in_stream.frames > 0 else None

        frame_buffer = {}
        last_bboxes = []
        key_idx = 0
        key_bboxes = None
        target_idx = 0
        fib_a, fib_b = 1, 2

        raw_detections = {}
        transitions = [0]

        def next_fib_skip() -> int:
            nonlocal fib_a, fib_b
            skip = fib_a
            fib_a, fib_b = fib_b, fib_a + fib_b
            if fib_b > self.max_skip:
                fib_a, fib_b = 1, 2
            return skip

        def binary_search_transition(bad_idx: int, k_bboxes: list[list[int]], new_bboxes: list[list[int]]) -> int:
            lo, hi = key_idx + 1, bad_idx
            result = bad_idx
            while lo <= hi:
                mid = (lo + hi) // 2
                if mid in frame_buffer:
                    f = frame_buffer[mid]
                    boxes = detect_bboxes(f, self.yolo_model, self.conf, self.iou)
                    self._yolo_count += 1
                    if boxes:
                        mid_bboxes = _group_and_merge_bboxes(boxes)
                        raw_detections[mid] = mid_bboxes
                        iou_old = _compute_multi_iou(mid_bboxes, k_bboxes)
                        iou_new = _compute_multi_iou(mid_bboxes, new_bboxes)
                        if iou_old >= iou_new:
                            lo = mid + 1
                        else:
                            result = mid
                            hi = mid - 1
                    else:
                        lo = mid + 1
                else:
                    hi = mid - 1
            return result

        pbar = tqdm(total=total_frames, desc="Stage 1: detection", unit="frame")
        frame_idx = 0
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            frame_buffer[frame_idx] = image

            if frame_idx >= target_idx:
                self._yolo_count += 1
                detected = detect_bboxes(image, self.yolo_model, self.conf, self.iou)
                merged = _group_and_merge_bboxes(detected) if detected else []

                if merged:
                    raw_detections[frame_idx] = merged
                    if key_bboxes is not None:
                        iou_val = _compute_multi_iou(merged, key_bboxes)
                        if iou_val < self.iou_threshold:
                            transition_idx = binary_search_transition(frame_idx, key_bboxes, merged)
                            transitions.append(transition_idx)
                            key_bboxes = merged
                            last_bboxes = merged
                            key_idx = transition_idx
                            fib_a, fib_b = 1, 2
                            target_idx = frame_idx + 1 + next_fib_skip()

                            for idx in list(frame_buffer.keys()):
                                if idx < key_idx:
                                    del frame_buffer[idx]
                            pbar.update(1)
                            frame_idx += 1
                            continue

                    key_bboxes = merged
                    last_bboxes = merged
                    key_idx = frame_idx
                    target_idx = frame_idx + 1 + next_fib_skip()

                    for idx in list(frame_buffer.keys()):
                        if idx < key_idx:
                            del frame_buffer[idx]
                else:
                    target_idx = frame_idx + 2

            pbar.update(1)
            frame_idx += 1

        pbar.close()
        total_frames = frame_idx
        transitions.append(total_frames)
        container.close()

        # Group transitions and perform segment-based stabilization.
        transitions = sorted(list(set(transitions)))

        for i in range(len(transitions) - 1):
            start = transitions[i]
            end = transitions[i+1] - 1

            # Group all raw bboxes in this segment into clusters based on center coordinates distance
            all_bboxes = []
            for idx in range(start, end + 1):
                if idx in raw_detections and raw_detections[idx]:
                    for b in raw_detections[idx]:
                        all_bboxes.append((idx, b))

            if all_bboxes:
                # Group bboxes whose centers are within 100 pixels of each other
                clusters = []
                for idx, b in all_bboxes:
                    cx = (b[0] + b[2]) / 2
                    cy = (b[1] + b[3]) / 2

                    found_cluster = False
                    for cluster in clusters:
                        ccx = sum((x[1][0] + x[1][2])/2 for x in cluster) / len(cluster)
                        ccy = sum((x[1][1] + x[1][3])/2 for x in cluster) / len(cluster)
                        if abs(cx - ccx) < 100 and abs(cy - ccy) < 100:
                            cluster.append((idx, b))
                            found_cluster = True
                            break
                    if not found_cluster:
                        clusters.append([(idx, b)])

                # Keep only clusters that appear in at least 20% of the unique frame detections in this segment
                unique_frames_with_detections = len(set(idx for idx, _ in all_bboxes))
                min_occurrences = max(1, int(0.2 * unique_frames_with_detections))

                stable_bboxes_for_segment = []
                cluster_ranges = []
                for cluster in clusters:
                    unique_frames_in_cluster = len(set(idx for idx, _ in cluster))
                    if unique_frames_in_cluster >= min_occurrences:
                        widths = [x[1][2] - x[1][0] for x in cluster]
                        heights = [x[1][3] - x[1][1] for x in cluster]

                        rough_w = np.median(widths)
                        rough_h = np.median(heights)

                        # Filter out size outliers within this cluster
                        filtered_cluster = [
                            x[1] for x in cluster
                            if 0.5 * rough_w <= (x[1][2] - x[1][0]) <= 2.0 * rough_w
                            and 0.5 * rough_h <= (x[1][3] - x[1][1]) <= 2.0 * rough_h
                        ]
                        if not filtered_cluster:
                            filtered_cluster = [x[1] for x in cluster]

                        widths = [b[2] - b[0] for b in filtered_cluster]
                        heights = [b[3] - b[1] for b in filtered_cluster]
                        cxs = [(b[0] + b[2]) / 2 for b in filtered_cluster]
                        cys = [(b[1] + b[3]) / 2 for b in filtered_cluster]

                        median_w = int(np.median(widths))
                        median_h = int(np.median(heights))
                        median_cx = int(np.median(cxs))
                        median_cy = int(np.median(cys))

                        stable_bbox = [
                            median_cx - median_w // 2,
                            median_cy - median_h // 2,
                            median_cx + median_w // 2,
                            median_cy + median_h // 2
                        ]
                        stable_bboxes_for_segment.append(stable_bbox)

                        # Calculate active range for this cluster
                        detected_indices = sorted(list(set(idx for idx, _ in cluster)))
                        first_idx = detected_indices[0]
                        last_idx = detected_indices[-1]
                        gaps = [detected_indices[j+1] - detected_indices[j] for j in range(len(detected_indices)-1)]
                        max_gap = max(gaps) if gaps else 10

                        cluster_ranges.append((
                            max(start, first_idx - max_gap),
                            min(end, last_idx + max_gap),
                            stable_bbox
                        ))

                # Initialize empty lists for the segment
                for idx in range(start, end + 1):
                    self._precalculated_bboxes[idx] = []

                # Populate only active frames for each cluster
                for c_start, c_end, bbox in cluster_ranges:
                    for idx in range(c_start, c_end + 1):
                        self._precalculated_bboxes[idx].append(bbox)
            else:
                # No detections in segment - carry forward last_bboxes or set empty list
                for idx in range(start, end + 1):
                    self._precalculated_bboxes[idx] = last_bboxes

    def __call__(self, frame: np.ndarray, frame_idx: int) -> list[list[int]]:
        return self._precalculated_bboxes.get(frame_idx, [])


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _detections_to_json(
    detections: dict[int, list[list[int]]], video_path: str
) -> dict:
    """Convert an in-memory detections dict to a JSON-serialisable structure.

    Args:
        detections: Mapping ``frame_idx -> list of [x1, y1, x2, y2]``.
        video_path: Original video path (stored for traceability).

    Returns:
        Dictionary ready for ``json.dump``.
    """
    frames = {}
    for idx, bboxes in detections.items():
        frames[str(idx)] = bboxes
    return {
        "video": os.path.basename(video_path),
        "total_frames": max(detections.keys()) + 1 if detections else 0,
        "frames": frames,
    }


def _json_to_detections(data: dict) -> dict[int, list[list[int]]]:
    """Load a detections JSON back into an in-memory dict.

    Args:
        data: Parsed JSON (as returned by :func:`_detections_to_json`).

    Returns:
        Mapping ``frame_idx -> list of [x1, y1, x2, y2]``.
    """
    frames = {}
    for key, bboxes in data.get("frames", {}).items():
        frames[int(key)] = bboxes
    return frames


def save_detections(
    detections: dict[int, list[list[int]]],
    output_path: str,
    video_path: str,
) -> None:
    """Persist detection results to a JSON file.

    Args:
        detections: Mapping ``frame_idx -> list of [x1, y1, x2, y2]``.
        output_path: Destination file path (will be created / overwritten).
        video_path: Original video path (stored for traceability).
    """
    payload = _detections_to_json(detections, video_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(
        "Saved %d frame entries to %s (%d total frames)",
        len(detections),
        output_path,
        payload["total_frames"],
    )


def load_detections(input_path: str) -> dict[int, list[list[int]]]:
    """Load detection results from a JSON file.

    Args:
        input_path: Path to a JSON file previously written by
            :func:`save_detections`.

    Returns:
        Mapping ``frame_idx -> list of [x1, y1, x2, y2]``.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the JSON structure is invalid.
    """
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    detections = _json_to_detections(data)
    logger.info(
        "Loaded %d frame entries from %s", len(detections), input_path
    )
    return detections


# ---------------------------------------------------------------------------
# Detection callable builder
# ---------------------------------------------------------------------------


def build_detection_fn(
    video_path: str,
    yolo_model,
    detect_mode: str,
    conf: float,
    iou: float,
    scene_threshold: float = 15.0,
    max_skip: int = 100,
    bbox: list[int] | None = None,
) -> Callable[[np.ndarray, int], list[list[int]]]:
    """Construct the per-frame detection callable for the given mode.

    Args:
        video_path: Path to the input video (needed by scene/smart modes).
        yolo_model: Loaded Ultralytics YOLO model.
        detect_mode: One of ``"per_frame"``, ``"scene"``, ``"smart"``.
        conf: YOLO confidence threshold.
        iou: YOLO NMS IoU threshold.
        scene_threshold: Content-detector sensitivity (scene mode only).
        max_skip: Maximum Fibonacci skip (smart mode only).
        bbox: Optional static ``[x1, y1, x2, y2]`` — overrides YOLO entirely.

    Returns:
        Callable ``(frame_ndarray, frame_idx) -> list[list[int]]``.
    """
    if bbox:
        return lambda _frame, _idx: [list(bbox)]

    if detect_mode == "scene":
        logger.info("Scene detection mode: pre-computing scene boundaries ...")
        return SmartTracker(
            SceneDetector(
                video_path, yolo_model, conf, iou, scene_threshold
            )
        )

    if detect_mode == "smart":
        logger.info(
            "Smart detection mode: Fibonacci frame skipping (max_skip=%d)",
            max_skip,
        )
        return SmartDetector(
            video_path, yolo_model, conf, iou, max_skip=max_skip
        )

    return SmartTracker(
        lambda f, _: detect_bboxes(f, yolo_model, conf, iou)
    )


# ---------------------------------------------------------------------------
# Stage 1 — full-video detection pass
# ---------------------------------------------------------------------------


def run_detection_stage(
    video_path: str,
    yolo_model,
    detect_mode: str,
    conf: float,
    iou: float,
    scene_threshold: float = 15.0,
    max_skip: int = 100,
    bbox: list[int] | None = None,
) -> dict[int, list[list[int]]]:
    """Run the full detection pass over the entire video.

    Reads every frame (or a smart subset), runs YOLO + normalisation,
    and returns a per-frame bbox mapping.  No inpainting is performed.

    Args:
        video_path: Path to the input video.
        yolo_model: Loaded Ultralytics YOLO model.
        detect_mode: One of ``"per_frame"``, ``"scene"``, ``"smart"``.
        conf: YOLO confidence threshold.
        iou: YOLO NMS IoU threshold.
        scene_threshold: Content-detector sensitivity (scene mode only).
        max_skip: Maximum Fibonacci skip (smart mode only).
        bbox: Optional static ``[x1, y1, x2, y2]`` — skip YOLO entirely.

    Returns:
        Mapping ``frame_idx -> list of [x1, y1, x2, y2]``.
    """
    detection_fn = build_detection_fn(
        video_path, yolo_model, detect_mode, conf, iou,
        scene_threshold, max_skip, bbox,
    )

    # SmartDetector pre-computes everything in __init__ (progress bar shown there).
    if isinstance(detection_fn, SmartDetector):
        return dict(detection_fn._precalculated_bboxes)

    # For other modes we iterate frame-by-frame.
    container = av.open(video_path)
    in_stream = container.streams.video[0]
    total_frames = in_stream.frames if in_stream.frames > 0 else None

    detections: dict[int, list[list[int]]] = {}
    pbar = tqdm(total=total_frames, desc="Stage 1: detection", unit="frame")

    for frame_idx, frame in enumerate(container.decode(video=0)):
        image = frame.to_ndarray(format="rgb24")
        bboxes = detection_fn(image, frame_idx)
        if bboxes:
            detections[frame_idx] = bboxes
        pbar.update(1)

    pbar.close()
    container.close()

    logger.info(
        "Detection pass complete: %d/%d frames have detections",
        len(detections),
        total_frames or "?",
    )
    return detections


