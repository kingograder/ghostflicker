import json
import logging
import os

import av
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def detect_bboxes(frame, model, conf, iou):
    res = model(frame, verbose=False, conf=conf, iou=iou)
    return (
        res[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
        if res and len(res[0].boxes) > 0
        else []
    )


def _merge_bboxes(boxes):
    if not boxes:
        return []
    arr = np.array(boxes)
    return [
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 2].max()),
        int(arr[:, 3].max()),
    ]


def _compute_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(a[2] - a[0], 1) * max(a[3] - a[1], 1)
    area_b = max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
    return inter / (area_a + area_b - inter)


def _compute_multi_iou(list_a, list_b):
    if not list_a or not list_b:
        return 0.0
    return sum(max(_compute_iou(a, b) for b in list_b) for a in list_a) / len(list_a)


def _group_and_merge_bboxes(boxes, gap=100):
    # Group overlapping boxes and merge them
    if not boxes:
        return []
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        parent[i] = find(parent[i]) if parent[i] != i else i
        return parent[i]

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if min(a[2] + gap, b[2] + gap) > max(a[0] - gap, b[0] - gap) and min(
                a[3] + gap, b[3] + gap
            ) > max(a[1] - gap, b[1] - gap):
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    return [_merge_bboxes(g) for g in groups.values()]


class PerFrameDetector:
    # Run YOLO on every frame and merge results
    def __init__(self, model, conf, iou):
        self.model, self.conf, self.iou = model, conf, iou

    def __call__(self, frame, _):
        det = detect_bboxes(frame, self.model, self.conf, self.iou)
        return [_merge_bboxes(det)] if det else []


class SceneDetector:
    # Run YOLO only on scene boundaries detected by pyscenedetect
    def __init__(self, video_path, model, conf, iou, threshold=15.0):
        self.model, self.conf, self.iou = model, conf, iou
        self.bounds = self._get_scenes(video_path, threshold)
        self.b_idx, self._last_bbox = 0, None

    @staticmethod
    def _get_scenes(path, threshold):
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        manager.detect_scenes(open_video(path, backend="pyav"))
        return (
            sorted({s.get_frames() for s, _ in manager.get_scene_list()})
            if manager.get_scene_list()
            else []
        )

    def __call__(self, frame, idx):
        if idx == 0 or (
            self.b_idx < len(self.bounds) and idx >= self.bounds[self.b_idx]
        ):
            if idx != 0:
                self.b_idx += 1
            det = detect_bboxes(frame, self.model, self.conf, self.iou)
            if det:
                self._last_bbox = _merge_bboxes(det)
        return [self._last_bbox] if self._last_bbox else []


class SmartDetector:
    # Fibonacci-skip scanning with binary search for transitions and clustering
    def __init__(self, video_path, model, conf, iou, iou_thresh=0.7, max_skip=100):
        self.model, self.conf, self.iou, self.iou_thresh, self.max_skip = (
            model,
            conf,
            iou,
            iou_thresh,
            max_skip,
        )
        self.bboxes = {}
        self._precalc(video_path)

    def _precalc(self, video_path):
        container = av.open(video_path)
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        total = stream.frames if stream.frames > 0 else None
        buf, raw, trans = {}, {}, [0]
        key_b, key_idx, target, fib_a, fib_b = None, 0, 0, 1, 2

        def next_fib():
            nonlocal fib_a, fib_b
            s = fib_a
            fib_a, fib_b = fib_b, fib_a + fib_b
            if fib_b > self.max_skip:
                fib_a, fib_b = 1, 2
            return s

        def bin_search(bad_idx, k_b, new_b):
            lo, hi, res = key_idx + 1, bad_idx, bad_idx
            while lo <= hi:
                mid = (lo + hi) // 2
                if mid in buf:
                    boxes = detect_bboxes(buf[mid], self.model, self.conf, self.iou)
                    if boxes:
                        m_b = _group_and_merge_bboxes(boxes)
                        raw[mid] = m_b
                        if _compute_multi_iou(m_b, k_b) >= _compute_multi_iou(
                            m_b, new_b
                        ):
                            lo = mid + 1
                        else:
                            res, hi = mid, mid - 1
                    else:
                        lo = mid + 1
                else:
                    hi = mid - 1
            return res

        pbar = tqdm(total=total, desc="Stage 1: Smart", unit="frame")
        for idx, frame in enumerate(container.decode(video=0)):
            buf[idx] = frame.to_ndarray(format="rgb24")
            if idx >= target:
                det = detect_bboxes(buf[idx], self.model, self.conf, self.iou)
                merged = _group_and_merge_bboxes(det) if det else []
                if merged:
                    raw[idx] = merged
                    if key_b and _compute_multi_iou(merged, key_b) < self.iou_thresh:
                        key_idx = bin_search(idx, key_b, merged)
                        trans.append(key_idx)
                        key_b, fib_a, fib_b = merged, 1, 2
                    else:
                        key_b, key_idx = merged, idx
                    target = idx + 1 + next_fib()
                    for k in list(buf.keys()):
                        if k < key_idx:
                            del buf[k]
                else:
                    target = idx + 2
            pbar.update(1)
        pbar.close()
        container.close()

        # Cluster and stabilize bounding boxes per segment
        trans = sorted(set(trans + [idx + 1]))
        for i in range(len(trans) - 1):
            start, end = trans[i], trans[i + 1] - 1
            all_b = [
                (i, b)
                for i in range(start, end + 1)
                if i in raw and raw[i]
                for b in raw[i]
            ]
            if not all_b:
                for i in range(start, end + 1):
                    self.bboxes[i] = key_b or []
                continue

            clusters = []
            for i, b in all_b:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                for c in clusters:
                    ccx = sum((x[1][0] + x[1][2]) / 2 for x in c) / len(c)
                    ccy = sum((x[1][1] + x[1][3]) / 2 for x in c) / len(c)
                    if abs(cx - ccx) < 100 and abs(cy - ccy) < 100:
                        c.append((i, b))
                        break
                else:
                    clusters.append([(i, b)])

            min_occ = max(1, int(0.2 * len(set(i for i, _ in all_b))))
            for i in range(start, end + 1):
                self.bboxes[i] = []

            for c in clusters:
                if len(set(i for i, _ in c)) < min_occ:
                    continue
                ws, hs = [x[1][2] - x[1][0] for x in c], [x[1][3] - x[1][1] for x in c]
                rw, rh = np.median(ws), np.median(hs)
                filt = [
                    x[1]
                    for x in c
                    if 0.5 * rw <= (x[1][2] - x[1][0]) <= 2.0 * rw
                    and 0.5 * rh <= (x[1][3] - x[1][1]) <= 2.0 * rh
                ] or [x[1] for x in c]

                mw = int(np.median([b[2] - b[0] for b in filt]))
                mh = int(np.median([b[3] - b[1] for b in filt]))
                mcx = int(np.median([(b[0] + b[2]) / 2 for b in filt]))
                mcy = int(np.median([(b[1] + b[3]) / 2 for b in filt]))
                bbox = [mcx - mw // 2, mcy - mh // 2, mcx + mw // 2, mcy + mh // 2]

                idxs = sorted(set(i for i, _ in c))
                gap = max([idxs[j + 1] - idxs[j] for j in range(len(idxs) - 1)] or [10])
                for i in range(max(start, idxs[0] - gap), min(end, idxs[-1] + gap) + 1):
                    self.bboxes[i].append(bbox)

    def __call__(self, frame, idx):
        return self.bboxes.get(idx, [])


def postprocess_detections(detections, max_gap=3, iou_threshold=0.3):
    if not detections:
        return {}

    frames = sorted(detections.keys())
    min_f, max_f = frames[0], frames[-1]

    # 1. Извлекаем последовательность bbox-ов
    seq = [
        detections.get(i, [])[0] if detections.get(i) else None
        for i in range(min_f, max_f + 1)
    ]

    # 2. Заполняем короткие пропуски (misses)
    last_valid = None
    gap = 0
    for i in range(len(seq)):
        if seq[i] is not None:
            last_valid = seq[i]
            gap = 0
        elif last_valid is not None and gap < max_gap:
            seq[i] = last_valid
            gap += 1
        else:
            last_valid = None

    # 3. Удаляем одиночные выбросы (глитчи), не трогая реальные перемещения
    for i in range(1, len(seq) - 1):
        if seq[i] is None or seq[i - 1] is None or seq[i + 1] is None:
            continue

        iou_prev = _compute_iou(seq[i], seq[i - 1])
        iou_next = _compute_iou(seq[i], seq[i + 1])
        iou_span = _compute_iou(seq[i - 1], seq[i + 1])

        # Условие глитча: текущий кадр не похож ни на предыдущий, ни на следующий,
        # но предыдущий и следующий похожи друг на друга.
        if (
            iou_prev < iou_threshold
            and iou_next < iou_threshold
            and iou_span > iou_threshold
        ):
            seq[i] = seq[i - 1]  # Заменяем глитч на стабильное предыдущее значение

    return {i: [seq[i]] for i in range(len(seq)) if seq[i] is not None}


def build_detection_fn(video_path, model, mode, conf, iou, scene_th, max_skip, bbox):
    if bbox:
        return lambda f, i: [list(bbox)]
    if mode == "scene":
        return SceneDetector(video_path, model, conf, iou, scene_th)
    if mode == "smart":
        return SmartDetector(video_path, model, conf, iou, max_skip=max_skip)
    return PerFrameDetector(model, conf, iou)


def run_detection_stage(
    video_path, model, mode, conf, iou, scene_th=15.0, max_skip=100, bbox=None
):
    fn = build_detection_fn(
        video_path, model, mode, conf, iou, scene_th, max_skip, bbox
    )
    if isinstance(fn, SmartDetector):
        return dict(fn.bboxes)

    container = av.open(video_path)
    if not container.streams.video:
        raise ValueError("No video stream")
    total = container.streams.video[0].frames or None

    raw_detections = {}
    pbar = tqdm(total=total, desc="Stage 1: detection", unit="frame")
    for i, frame in enumerate(container.decode(video=0)):
        b = fn(frame.to_ndarray(format="rgb24"), i)
        if b:
            raw_detections[i] = b
        pbar.update(1)
    pbar.close()
    container.close()

    return postprocess_detections(raw_detections, max_gap=5)


def save_detections(det, path, video_path):
    data = {
        "video": os.path.basename(video_path),
        "total_frames": max(det.keys()) + 1 if det else 0,
        "frames": {str(k): v for k, v in det.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d frames to %s", len(det), path)


def load_detections(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    det = {int(k): v for k, v in data.get("frames", {}).items()}
    logger.info("Loaded %d frames from %s", len(det), path)
    return det
