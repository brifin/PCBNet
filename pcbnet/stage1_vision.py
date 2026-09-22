"""Stage 1: component/text detection, OCR, wire skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any 
from .visual import run_vision_visualize, run_gt_visualize, run_wire_visualize
from .topo_graph import extract_wire_nets
from tqdm import tqdm

import cv2
import numpy as np

import logging
# 禁用 paddle DEBUG 和 WARNING 级别的日志
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

@dataclass
class DetBox:
    bbox: list[float]  # image xyxy
    label: str
    score: float = 1.0


@dataclass
class TextBox:
    bbox: list[float]
    content: str
    score: float = 1.0


@dataclass
class PinBox:
    xy: tuple[float, float]  # tip point in image coords
    score: float = 1.0
    number: str | None = None
    name: str | None = None


@dataclass
class VisionOut:
    height: int
    width: int
    components: list[DetBox] = field(default_factory=list)
    texts: list[TextBox] = field(default_factory=list)
    pins: list[PinBox] = field(default_factory=list)
    skeleton: np.ndarray | None = None
    wire_segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    img: np.ndarray | None = None
    image_path: str | Path | None = None


# Non-component detection classes (handled separately from components).
TEXT_LABEL = "text"
PIN_LABEL = "pin_tip"
ENDPOINT_LABEL = "wire_endpoint"

def _component_labels() -> set[str]:
    try:
        import yaml

        path = Path(__file__).resolve().parent.parent / "data" / "yolo_classes.yaml"
        names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names") or []
        return {str(n) for n in names if n not in (TEXT_LABEL, PIN_LABEL)}
    except Exception:
        return set()


def _read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _try_yolo(img: np.ndarray, weights: str) -> list[DetBox]:
    if not weights:
        return []

    from ultralytics import YOLO

    model = YOLO(weights)
    res = model.predict(img, verbose=False)[0]
    boxes = []
    names = res.names
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        boxes.append(DetBox(bbox=xyxy, label=str(names.get(cls, cls)), score=float(b.conf[0])))
    return boxes



def _nms(boxes: list[DetBox], iou_thr: float = 0.35) -> list[DetBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]), reverse=True)
    keep: list[DetBox] = []
    for box in boxes:
        x1, y1, x2, y2 = box.bbox
        area = max(1.0, (x2 - x1) * (y2 - y1))
        dup = False
        for other in keep:
            ox1, oy1, ox2, oy2 = other.bbox
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            oarea = max(1.0, (ox2 - ox1) * (oy2 - oy1))
            if inter / min(area, oarea) > iou_thr:
                dup = True
                break
        if not dup:
            keep.append(box)
    return keep


def _rule_components(img: np.ndarray, max_components: int = 400) -> list[DetBox]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    boxes: list[DetBox] = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 60 or area > 0.25 * w * h:
            continue
        if ww < 6 or hh < 6:
            continue
        aspect = max(ww / max(hh, 1), hh / max(ww, 1))
        if aspect > 12:
            continue
        if min(ww, hh) > 28 and aspect < 3.5:
            label = "Box"
        elif aspect > 2.2:
            label = "R"
        else:
            label = "C"
        boxes.append(DetBox(bbox=[x, y, x + ww, y + hh], label=label, score=0.3))
    return _nms(boxes)[:max_components]


_OCR_ENGINE = None
_OCR_ERROR_PRINTED = False


def _install_torch_stub() -> None:
    """Register a minimal fake `torch` so a torch-free process can import modelscope.

    Only used inside the paddle-only OCR process where real torch cannot be loaded
    (cudnn DLL clash). Provides just the names modelscope touches at import time.
    """
    import importlib.machinery as machinery
    import sys
    import types
    from unittest.mock import MagicMock

    def _spec(name: str, pkg: bool = False) -> machinery.ModuleSpec:
        spec = machinery.ModuleSpec(name, loader=None, is_package=pkg)
        return spec

    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0+stub"
    torch.__spec__ = _spec("torch", pkg=True)
    torch.__path__ = []  # type: ignore[attr-defined]
    torch.__getattr__ = lambda name: MagicMock(name=f"torch.{name}")  # type: ignore[attr-defined]
    mp = types.ModuleType("torch.multiprocessing")
    mp.__spec__ = _spec("torch.multiprocessing")
    mp.__getattr__ = lambda name: MagicMock(name=f"torch.multiprocessing.{name}")  # type: ignore[attr-defined]
    dist = types.ModuleType("torch.distributed")
    dist.__spec__ = _spec("torch.distributed")
    dist.is_available = lambda: False  # type: ignore[attr-defined]
    dist.is_initialized = lambda: False  # type: ignore[attr-defined]
    dist.get_rank = lambda *a, **k: 0  # type: ignore[attr-defined]
    dist.get_world_size = lambda *a, **k: 1  # type: ignore[attr-defined]
    dist.__getattr__ = lambda name: MagicMock(name=f"torch.distributed.{name}")  # type: ignore[attr-defined]
    torch.multiprocessing = mp  # type: ignore[attr-defined]
    torch.distributed = dist  # type: ignore[attr-defined]
    sys.modules["torch"] = torch
    sys.modules["torch.multiprocessing"] = mp
    sys.modules["torch.distributed"] = dist


def _get_ocr(lang: str = "en"):
    global _OCR_ENGINE

    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    from paddleocr import PaddleOCR
    
    _OCR_ENGINE = PaddleOCR(
        lang=lang,
        device="cuda:0",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # if _OCR_ENGINE is not None:
    #     return _OCR_ENGINE
    # # IMPORTANT: fully import paddle BEFORE paddleocr/paddlex. paddlex lazily
    # # re-imports paddle during model construction; if paddleocr is imported first
    # # paddle can be left partially initialized ("no attribute 'tensor'"). Importing
    # # paddle first completes paddle/__init__ (incl. monkey_patch) and caches it.
    # device = "cpu"
    # try:
    #     import paddle  # noqa: F401

    #     _ = paddle.tensor  # force full init
    #     if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
    #         device = "gpu"
    # except Exception:
    #     device = "cpu"

    # # paddlex unconditionally imports modelscope, which imports torch. torch's
    # # bundled cudnn DLL clashes with paddle's in a single process (Windows
    # # WinError 127). In this paddle-only OCR process real torch cannot load, so
    # # inject a lightweight stub to satisfy modelscope's import (models are cached,
    # # no download happens). Guarded: never runs when real torch is already loaded.
    # import sys as _sys

    # if "torch" not in _sys.modules:
    #     try:
    #         import torch  # noqa: F401
    #     except Exception:
    #         _install_torch_stub()

    # from paddleocr import PaddleOCR

    # try:
    #     _OCR_ENGINE = PaddleOCR(
    #         lang=lang,
    #         device="cuda:0",
    #         use_doc_orientation_classify=False,
    #         use_doc_unwarping=False,
    #         use_textline_orientation=False,
    #     )
    # except TypeError:
    #     try:
    #         _OCR_ENGINE = PaddleOCR(
    #             lang=lang,
    #             use_doc_orientation_classify=False,
    #             use_doc_unwarping=False,
    #             use_textline_orientation=False,
    #         )
    #     except TypeError:
    #         _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _OCR_ENGINE


def _box_from_poly(poly) -> list[float] | None:
    try:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        return [min(xs), min(ys), max(xs), max(ys)]
    except Exception:
        return None


def _parse_ocr_result(result) -> list[TextBox]:
    texts: list[TextBox] = []
    if result is None:
        return texts
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue
        if isinstance(page, dict) or hasattr(page, "get"):
            rec_texts = list(page.get("rec_texts") or page.get("rec_text") or [])
            rec_scores = list(page.get("rec_scores") or page.get("rec_score") or [1.0] * len(rec_texts))
            rec_polys = list(page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes") or [])
            for i, txt in enumerate(rec_texts):
                poly = rec_polys[i] if i < len(rec_polys) else [[0, 0], [1, 0], [1, 1], [0, 1]]
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                if isinstance(poly, (list, tuple)) and len(poly) == 4 and not isinstance(poly[0], (list, tuple)):
                    x1, y1, x2, y2 = [float(v) for v in poly]
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                else:
                    bbox = _box_from_poly(poly) or [0, 0, 1, 1]
                conf = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                texts.append(TextBox(bbox=bbox, content=str(txt).strip(), score=conf))
            continue
        if not isinstance(page, list):
            continue
        for item in page:
            if not item:
                continue
            box, payload = item[0], item[1]
            txt, conf = payload[0], payload[1]
            bbox = _box_from_poly(box)
            if bbox is None:
                continue
            texts.append(TextBox(bbox=bbox, content=str(txt).strip(), score=float(conf)))
    return [t for t in texts if t.content]


def _ocr_texts(img: np.ndarray, enabled: bool = True, lang: str = "en") -> list[TextBox]:
    global _OCR_ERROR_PRINTED
    if not enabled:
        return []
    try:
        ocr = _get_ocr(lang)
        if hasattr(ocr, "predict"):
            result = ocr.predict(img)
        else:
            result = ocr.ocr(img, cls=True)
        return _parse_ocr_result(result)
    except Exception as exc:
        if not _OCR_ERROR_PRINTED:
            print(f"[ocr] PaddleOCR unavailable: {exc}")
            _OCR_ERROR_PRINTED = True
        return []


def _ocr_best_string(crop: np.ndarray, lang: str = "en") -> tuple[str, float]:
    """OCR a crop upright and rotated 90 deg (for vertical labels); keep higher conf."""
    best_txt, best_conf = "", 0.0
    import cv2

    variants = [crop]
    if crop.shape[0] > crop.shape[1] * 1.3:  # tall crop -> likely vertical text
        variants.append(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
        variants.append(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE))
    for var in variants:
        for tb in _ocr_texts(var, enabled=True, lang=lang):
            if tb.score > best_conf and tb.content:
                best_txt, best_conf = tb.content, tb.score
    return best_txt, best_conf


def _ocr_on_boxes(img: np.ndarray, boxes: list[list[float]], lang: str = "en", pad: int = 3) -> list[TextBox]:
    """Crop each detected text box (padded), OCR upright + rotated, keep best."""
    h, w = img.shape[:2]
    out: list[TextBox] = []
    for bb in tqdm(boxes, desc="Text Box Infering: "):
        x1 = max(0, int(bb[0]) - pad)
        y1 = max(0, int(bb[1]) - pad)
        x2 = min(w, int(bb[2]) + pad)
        y2 = min(h, int(bb[3]) + pad)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        crop = img[y1:y2, x1:x2]
        txt, conf = _ocr_best_string(crop, lang=lang)
        if txt:
            out.append(TextBox(bbox=[x1, y1, x2, y2], content=txt, score=conf))
    return out


def _maybe_resize(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def _skeleton_wires(
    img: np.ndarray, max_side: int = 1600, max_segments: int = 4000
) -> tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]]]:
    work, scale = _maybe_resize(img, max_side)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thick = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    thin = cv2.subtract(bw, thick)
    try:
        from skimage.morphology import skeletonize

        skel_s = skeletonize(thin > 0).astype(np.uint8) * 255
    except Exception:
        skel_s = cv2.ximgproc.thinning(thin) if hasattr(cv2, "ximgproc") else thin
    h, w = img.shape[:2]
    skel = cv2.resize(skel_s, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else skel_s
    segs = _skel_to_segments(skel, max_segments=max_segments)
    return skel, segs


def _skel_to_segments(
    skel: np.ndarray, min_len: int = 6, max_segments: int = 4000
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if skel is None or not np.any(skel):
        return []
    dil = cv2.dilate(skel, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    segs = []
    for cnt in contours:
        if len(cnt) < min_len:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt[:: max(1, len(cnt) // 40)]]
        for i in range(len(pts) - 1):
            if abs(pts[i][0] - pts[i + 1][0]) + abs(pts[i][1] - pts[i + 1][1]) >= 2:
                segs.append((pts[i], pts[i + 1]))
        if len(segs) >= max_segments:
            break
    return segs[:max_segments]


def _try_unet_wires(
    img: np.ndarray, weights: str, max_side: int, max_segments: int
) -> tuple[np.ndarray, list] | None:
    if not weights or not Path(weights).exists():
        return None
    try:
        from skimage.morphology import skeletonize

        from .unet import predict_mask

        mask = predict_mask(weights, img, size=min(512, max(256, max_side)))
        skel = skeletonize(mask > 0).astype(np.uint8) * 255
        return skel, _skel_to_segments(skel, max_segments=max_segments)
    except Exception:
        return None


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _cover_frac(inner: list[float], outer: list[float]) -> float:
    """Fraction of `inner`'s area that is covered by `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


# Symbol-like classes where a box fully covered by printed text is a misdetection.
# Power/label classes are inherently text-like and excluded from that filter.
_TEXTLIKE_EXEMPT = {"net_label", "gnd", "vcc"}

def _clean_components(
    comps: list[DetBox],
    text_boxes: list[list[float]],
    text_cover_thr: float = 0.7,
    dedup_iou: float = 0.55,
    contain_thr: float = 0.8,
    cross_class_iou_thr: float = 0.65,
    comp_score_thr: float = 0.0,
    text_box_score_thr: float = 0.0,
) -> list[DetBox]:
    """
    规则更新：
    1. 剔除被文字大面积覆盖的器件框
    2. 【修复】相同label分组预处理：只要一个框大面积包含另一个，删除面积更小的框
    3. 不同label：bbox高度重合(IOU >= cross_class_iou_thr)，只保留置信度最高的框
    """
    # ========== 过滤低置信元器件框 ==========
    comps = [c for c in comps if c.score >= comp_score_thr]
    if not comps:
        return []
    # ========== 过滤低置信YOLO文本框 ==========
    filtered_text_boxes: list[list[float]] = []
    for tb in text_boxes:
        if tb[4] >= text_box_score_thr:
            filtered_text_boxes.append(tb)
    text_boxes = filtered_text_boxes

    # 1) 剔除几乎完全被文本覆盖的器件框（豁免_net_label,gnd,vcc）
    kept: list[DetBox] = []
    for c in comps:
        if c.label not in _TEXTLIKE_EXEMPT and text_boxes:
            max_cover = max((_cover_frac(c.bbox, tb) for tb in text_boxes), default=0.0)
            if max_cover >= text_cover_thr:
                continue
        kept.append(c)
    if not kept:
        return []

    def box_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    # ======================
    # StepA: 同label分组预处理，剔除互相包含的小框
    # ======================
    from collections import defaultdict
    label_groups = defaultdict(list)
    for box in kept:
        label_groups[box.label].append(box)

    deduped_same_label = []
    for label, group_boxes in label_groups.items():
        # 同label组：按面积降序排序，大框在前
        group_boxes.sort(key=lambda b: box_area(b.bbox), reverse=True)
        group_final = []
        for box in group_boxes:
            dup = False
            for exist in group_final:
                # 判断：box被exist包含 OR exist被box包含
                cover_box_in_exist = _cover_frac(box.bbox, exist.bbox)
                cover_exist_in_box = _cover_frac(exist.bbox, box.bbox)
                iou_val = _bbox_iou(box.bbox, exist.bbox)
                if iou_val >= dedup_iou or cover_box_in_exist >= contain_thr or cover_exist_in_box >= contain_thr:
                    dup = True
                    break
            if not dup:
                group_final.append(box)
        deduped_same_label.extend(group_final)

    # ======================
    # StepB: 全局按score降序，处理跨类别高IOU冲突
    # ======================
    deduped_same_label.sort(key=lambda d: d.score, reverse=True)

    final: list[DetBox] = []
    for c in deduped_same_label:
        dup = False
        for f in final:
            iou_val = _bbox_iou(c.bbox, f.bbox)
            if c.label == f.label:
                # 同label已经预处理完，这里直接跳过同类判断
                continue
            else:
                # 不同label，高度重合，丢弃低分框
                if iou_val >= cross_class_iou_thr:
                    dup = True
                    break
        if not dup:
            final.append(c)
    return final



def _split_detections(dets: list[DetBox]) -> tuple[list[DetBox], list[list[float]], list[PinBox], list[PinBox]]:
    comp_labels = _component_labels()
    comps: list[DetBox] = []
    text_boxes: list[list[float]] = []
    pins: list[PinBox] = []
    endpoints: list[PinBox] = []
    for d in dets:
        if d.label == TEXT_LABEL:
            text_boxes.append(d.bbox + [d.score])
        elif d.label == PIN_LABEL:
            cx = (d.bbox[0] + d.bbox[2]) / 2.0
            cy = (d.bbox[1] + d.bbox[3]) / 2.0
            pins.append(PinBox(xy=(cx, cy), score=d.score))
        elif d.label == ENDPOINT_LABEL:
            cx = (d.bbox[0] + d.bbox[2]) / 2.0
            cy = (d.bbox[1] + d.bbox[3]) / 2.0
            endpoints.append(PinBox(xy=(cx, cy), score=d.score))
        elif not comp_labels or d.label in comp_labels:
            comps.append(d)
    return comps, text_boxes, pins, endpoints


def _load_cached_ocr(image_path: str | Path, cfg: dict[str, Any]) -> list[TextBox] | None:
    """Return cached OCR text boxes for this image, or None if no cache exists."""
    try:
        from .config import project_root
        from .ocr_cache import load_json, ocr_cache_file

        out_root = project_root() / (cfg.get("output_root") or "outputs")
        data = load_json(ocr_cache_file(out_root, image_path))
        if data is None:
            return None
        return [
            TextBox(bbox=[float(v) for v in t["bbox"]], content=str(t.get("content") or ""), score=float(t.get("score", 1.0)))
            for t in data
            if t.get("content")
        ]
    except Exception:
        return None


def extract_vision(image_path: str | Path, cfg: dict[str, Any] | None = None) -> VisionOut:
    cfg = cfg or {}
    img = _read_image(image_path)
    h, w = img.shape[:2]
    models = cfg.get("models") or {}
    vision_cfg = cfg.get("vision") or {}
    max_side = int(vision_cfg.get("max_side", 1600))
    max_components = int(vision_cfg.get("max_components", 400))
    max_segments = int(vision_cfg.get("max_segments", 4000))
    ocr_cfg = cfg.get("ocr") or {}
    ocr_enabled = bool(ocr_cfg.get("enabled", True))
    lang = str(ocr_cfg.get("lang") or "en")

    dets = _try_yolo(img, models.get("yolo_weights") or "")
    comps, text_boxes, pins, endpoints = _split_detections(dets)
    comps = _clean_components(comps, text_boxes)

    if not comps:
        # No detector: rule-based component fallback (weak, low score expected).
        work, scale = _maybe_resize(img, max_side)
        comps = _rule_components(work, max_components=max_components)
        if scale < 1.0:
            for box in comps:
                box.bbox = [v / scale for v in box.bbox]

    texts = _load_cached_ocr(image_path, cfg) if ocr_enabled else []
    if texts is None:
        # No cache: OCR in-process (only works in a paddle-only process; when
        # torch/YOLO is loaded this raises a cudnn DLL clash and returns []).
        if text_boxes:
            texts = _ocr_on_boxes(img, text_boxes, lang=lang) if ocr_enabled else []
        else:
            ocr_img, _ = _maybe_resize(img, max_side)
            texts = _ocr_texts(ocr_img, enabled=ocr_enabled, lang=lang)
            if ocr_img.shape[0] != h or ocr_img.shape[1] != w:
                sx, sy = w / ocr_img.shape[1], h / ocr_img.shape[0]
                for t in texts:
                    t.bbox = [t.bbox[0] * sx, t.bbox[1] * sy, t.bbox[2] * sx, t.bbox[3] * sy]

    wired = _try_unet_wires(img, models.get("unet_weights") or "", max_side, max_segments)
    if wired is None:
        skel, segs = _skeleton_wires(img, max_side=max_side, max_segments=max_segments)
    else:
        skel, segs = wired

    # 可视化
    run_vision_visualize(image_path, img, comps, texts, skel, pins, endpoints)
    run_gt_visualize(image_path, img)

    out = extract_wire_nets(img, image_path, pins, comps, endpoints)
    run_wire_visualize(out, img, image_path)

    return VisionOut(height=h, width=w, components=comps, texts=texts, pins=pins, skeleton=skel, wire_segments=segs, img=img, image_path=image_path)
