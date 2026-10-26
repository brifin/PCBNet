"""Stage 1: component/text detection, OCR, wire skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any 
from .visual import run_vision_visualize, run_gt_visualize, run_wire_visualize, draw_pcb_pin_net
from .topo_graph import extract_wire_nets
from tqdm import tqdm
from .skele_wire import extract_nets

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
    endpoints: list[PinBox] = field(default_factory=list)   # wire_endpoint/pin_tip 检测结果
    skeleton: np.ndarray | None = None
    wire_segments: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    # topo_graph.extract_wire_nets 的导线网络重建结果，stage2 优先用它生成 nets
    wire_nets: dict = field(default_factory=dict)
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


_YOLO_DEVICE: str | None = None


def _resolve_yolo_device() -> str:
    """显式选择推理设备：能用 CUDA 就用 GPU，否则 CPU（并静音 torch 的驱动告警）。

    环境变量 PCB_YOLO_DEVICE 可强制指定（如 cpu / 0）。评测机驱动过旧时
    （镜像 torch 是 cu130，驱动仅支持 CUDA 12.9），torch.cuda 会初始化失败，
    这里安全回退 CPU，不抛异常、不刷 stderr。
    """
    global _YOLO_DEVICE
    if _YOLO_DEVICE is not None:
        return _YOLO_DEVICE
    import os

    forced = os.environ.get("PCB_YOLO_DEVICE", "").strip()
    if forced:
        _YOLO_DEVICE = forced
    else:
        try:
            import warnings

            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _YOLO_DEVICE = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            _YOLO_DEVICE = "cpu"
    print(f"[vision] YOLO device = {_YOLO_DEVICE}", flush=True)
    return _YOLO_DEVICE

def _try_yolo(img: np.ndarray, weights: str, conf: float = 0.4) -> list[DetBox]:
    if not weights:
        return []

    from ultralytics import YOLO

    model = YOLO(weights)
    try:
        res = model.predict(img, conf=conf, device=_resolve_yolo_device(), verbose=False)[0]
    except Exception as exc:  # noqa: BLE001 设备不可用时回退 CPU 再试
        print(f"[vision] 推理设备失败({exc})，回退 CPU", flush=True)
        global _YOLO_DEVICE
        _YOLO_DEVICE = "cpu"
        res = model.predict(img, conf=conf, device=_resolve_yolo_device(), verbose=False)[0]

    boxes = []
    names = res.names
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        boxes.append(DetBox(bbox=xyxy, label=str(names.get(cls, cls)), score=float(b.conf[0])))

    return boxes


def _try_yolo_v2(img: np.ndarray, weights: str, comp_conf: float = 0.4, pin_conf: float = 0.05) -> list[DetBox]:
    if not weights:
        return []

    from ultralytics import YOLO

    model = YOLO(weights)
    try:
        comp_res = model.predict(img, conf=comp_conf, device=_resolve_yolo_device(), verbose=False)[0]
        pin_res = model.predict(img, conf=pin_conf, device=_resolve_yolo_device(), verbose=False)[0]
    except Exception as exc:  # noqa: BLE001 设备不可用时回退 CPU 再试
        print(f"[vision] 推理设备失败({exc})，回退 CPU", flush=True)
        global _YOLO_DEVICE
        _YOLO_DEVICE = "cpu"
        comp_res = model.predict(img, conf=comp_conf, device=_resolve_yolo_device(), verbose=False)[0]
        pin_res = model.predict(img, conf=pin_conf, device=_resolve_yolo_device(), verbose=False)[0]

    boxes = []
    names = comp_res.names
    for b in comp_res.boxes:
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        if cls not in [14, 15]:
            boxes.append(DetBox(bbox=xyxy, label=str(names.get(cls, cls)), score=float(b.conf[0])))
    
    for b in pin_res.boxes:
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        if cls in [0, 1]:
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


def _ocr_device() -> str:
    """device 字符串随 paddleocr 版本而异：3.x 用 "gpu:N"，2.x 用 "cuda:N"。

    可用环境变量 PCBOCR_DEVICE 显式指定（如评测容器内装的是 CPU 版 paddle，
    设 PCBOCR_DEVICE=cpu 避免与 torch 的 CUDA 库冲突）。
    """
    import os

    env = os.environ.get("PCBOCR_DEVICE", "").strip()
    if env:
        return env
    try:
        import paddleocr

        ver = str(getattr(paddleocr, "__version__", ""))
    except Exception:
        ver = ""
    return "gpu:0" if ver.startswith("3") else "cuda:0"


def _get_ocr(lang: str = "en"):
    global _OCR_ENGINE

    if _OCR_ENGINE is not None:
        return _OCR_ENGINE

    from paddleocr import PaddleOCR

    # det_limit_side_len：paddle 默认 960，会把大图缩得很小导致密集小字检不出
    # （实测 10gt case2：960 → 6 条文本；1600 → 46 条；2560 → 51 条）。可按需调大。
    import os as _os

    try:
        _det_limit = int(_os.environ.get("PCBOCR_DET_LIMIT", "2560"))
    except ValueError:
        _det_limit = 2560
    _OCR_ENGINE = PaddleOCR(
        lang=lang,
        device=_ocr_device(),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        det_limit_side_len=_det_limit,
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


def _full_image_ocr(image_path: str | Path, lang: str = "en") -> list[TextBox]:

    """每次推理都做一次**全图 OCR**（不读缓存）。

    torch 已加载时（主推理进程），paddlepaddle-gpu 与 torch 的 CUDA 库冲突，
    改用独立子进程跑 OCR（pcbnet/_ocr_worker.py，可用 PCB_OCR_PYTHON 指定解释器）；
    否则（纯 paddle 进程）直接进程内识别。
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    import torch  # noqa: F401  仅探测是否已加载

    if "torch" in sys.modules:
        worker = Path(__file__).resolve().parent / "_ocr_worker.py"
        interpreter = os.environ.get("PCB_OCR_PYTHON") or sys.executable
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            out_path = fh.name
        try:
            env = dict(os.environ)
            env.setdefault("PCBOCR_LANG", lang)
            subprocess.run(
                [interpreter, str(worker), str(image_path), out_path],
                check=True, env=env, timeout=1800,
            )
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        finally:
            try:
                Path(out_path).unlink()
            except OSError:
                pass
        return [TextBox(bbox=list(d["bbox"]), content=d["content"], score=float(d["score"])) for d in data]
    img = _read_image(image_path)
    return _ocr_texts(img, enabled=True, lang=lang)


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
) -> tuple[np.ndarray, list, list[tuple[float, float]]] | None:
    """双通道 UNet：导线 mask → 骨架线段；连接点 mask → 小连通斑的质心。"""
    if not weights or not Path(weights).exists():
        return None
    try:
        from skimage.morphology import skeletonize

        from .unet import predict_wire_points

        wmask, pmask = predict_wire_points(weights, img, size=min(512, max(256, max_side)))
        skel = skeletonize(wmask > 0).astype(np.uint8) * 255
        points: list[tuple[float, float]] = []
        nlab, lab, stats, cent = cv2.connectedComponentsWithStats(pmask, 8)
        for i in range(1, nlab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 6 <= area <= 600:      # 太小=噪声，太大=不是点
                points.append((float(cent[i, 0]), float(cent[i, 1])))
        # NMS/去重：8px 内的点只留一个
        points = sorted(points, key=lambda q: (-int(pmask[int(q[1]), int(q[0])]), q[0]))
        dedup: list[tuple[float, float]] = []
        for (px, py) in points:
            if any((px - qx) ** 2 + (py - qy) ** 2 < 64 for qx, qy in dedup):
                continue
            dedup.append((px, py))
        return skel, _skel_to_segments(skel, max_segments=max_segments), dedup
    except Exception:
        return None


def _detect_ic_pins_crop(
    img: np.ndarray, comps: list[DetBox], weights: str,
    conf: float = 0.15, margin: float = 0.3, dedup_px: float = 8.0,
) -> list[PinBox]:
    """两阶段检测：对每个 ic（box）元件裁剪放大（含引脚范围）再跑一次 YOLO 检 pin_tip。

    crop 范围 = box 外扩 margin*max(box宽,box高)（夹在图像内）；crop 放大后引脚更清晰。
    结果按 8px 去重（保留置信度高的），与全图检测互不干扰。
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    h, w = img.shape[:2]
    found: list[PinBox] = []
    for comp in comps:
        if comp.label != "ic":
            continue
        x1, y1, x2, y2 = (float(v) for v in comp.bbox)
        m = margin * max(x2 - x1, y2 - y1, 1.0)
        cx1, cy1 = max(0.0, x1 - m), max(0.0, y1 - m)
        cx2, cy2 = min(float(w), x2 + m), min(float(h), y2 + m)
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            continue
        crop = img[int(cy1):int(cy2), int(cx1):int(cx2)]
        try:
            res = model.predict(crop, imgsz=640, conf=conf, verbose=False)[0]
        except Exception:
            continue
        if res.boxes is None:
            continue
        for b, cl, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            if int(cl) != 15:      # pin_tip
                continue
            px = (b[0] + b[2]) / 2 + cx1
            py = (b[1] + b[3]) / 2 + cy1
            found.append(PinBox(xy=(px, py), score=float(cf)))
    # NMS/去重：8px 内只留置信度最高的
    found.sort(key=lambda p: -float(getattr(p, "score", 0.5)))
    keep: list[PinBox] = []
    for p in found:
        if any((p.xy[0] - q.xy[0]) ** 2 + (p.xy[1] - q.xy[1]) ** 2 < dedup_px * dedup_px for q in keep):
            continue
        keep.append(p)
    return keep


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

def _nms_components(comps: list[DetBox], iou_thr: float = 0.5) -> list[DetBox]:
    """按类 NMS：同类元件框 IoU>阈值只留置信度最高者（防重复检测框）。"""
    if not comps:
        return comps
    from collections import defaultdict

    by_label: dict[str, list[DetBox]] = defaultdict(list)
    for c in comps:
        by_label[c.label].append(c)
    out: list[DetBox] = []
    for lst in by_label.values():
        lst.sort(key=lambda b: -float(b.score))
        keep = []
        for b in lst:
            drop = False
            for k in keep:
                if _bbox_iou(b.bbox, k.bbox) > iou_thr:
                    drop = True
                    break
            if not drop:
                keep.append(b)
        out.extend(keep)
    return out


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


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU计算，DetBox.bbox格式：[x1,y1,x2,y2]"""
    x1a, y1a, x2a, y2a = box_a
    x1b, y1b, x2b, y2b = box_b
    # 交集
    x1 = max(x1a, x1b)
    y1 = max(y1a, y1b)
    x2 = min(x2a, x2b)
    y2 = min(y2a, y2b)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    # 两个框面积
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union_area = area_a + area_b - inter_area
    if union_area < 1e-8:
        return 0.0
    return inter_area / union_area



def extract_vision(
    image_path: str | Path, cfg: dict[str, Any] | None = None, visualize: bool = True
) -> VisionOut:
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

    # dets = _try_yolo_v2(img, models.get("yolo_weights") or "", comp_conf=float(vision_cfg.get("yolo_comp_conf", 0.05)), pin_conf=float(vision_cfg.get("yolo_pin_conf", 0.2)))
    # comps, text_boxes, pins, endpoints = _split_detections(dets)
    # comps = _clean_components(comps, text_boxes, cross_class_iou_thr=0.2)
    # comps = _nms_components(comps, iou_thr=float(vision_cfg.get("comp_nms_iou", 0.01)))

    # ========== 迭代检测逻辑（直接粘贴到你原有代码后面） ==========
    # 复制原图作为迭代图像，保护原始img不被涂白修改
    img_iter = img.copy()

    total_rounds = 0
    total_new_comps = 0
    comps = []  # 保存已经识别过的器件框，用来判断是否新增
    yolo_confs = [None, 0.4, 0.25]
    while True:
        total_rounds += 1
        # 复用你原有的YOLO推理+后处理流水线
        dets = _try_yolo(img_iter, models.get("yolo_weights") or "", conf=yolo_confs[min(total_rounds, len(yolo_confs)-1)])
        if total_rounds == 1:
            comp, text_boxes, pins, endpoints = _split_detections(dets)
        else:
            comp, _, _, _ = _split_detections(dets)
        comp = _clean_components(comp, text_boxes)
        comp = _nms_components(comp, iou_thr=0.01)

        # 判断本轮哪些器件是【新增】：与历史所有框IoU都小于阈值才算新目标
        new_in_round = []
        for box in comp:
            new_flag = True
            for seen_box in comps:
                iou = compute_iou(box.bbox, seen_box.bbox)
                if iou > 0.01:
                    new_flag = False
                    break
            if new_flag:
                new_in_round.append(box)

        # 本轮无新增器件，终止迭代
        if len(new_in_round) == 0:
            break

        if total_rounds > 1:
            total_new_comps += len(new_in_round)
        comps.extend(new_in_round)

        # 涂白本轮新检测出来的器件bbox
        h, w = img_iter.shape[:2]
        for b in new_in_round:
            x1, y1, x2, y2 = b.bbox
            # 浮点坐标转像素，clip边界防止越界
            x1i = max(0, int(round(x1 - 0)))
            y1i = max(0, int(round(y1 - 0)))
            x2i = min(w, int(round(x2 + 0)))
            y2i = min(h, int(round(y2 + 0)))

            # 填充白色矩形
            cv2.rectangle(img_iter, (x1i, y1i), (x2i, y2i), (255,255,255), thickness=-1)

        # import os
        # save_path = os.path.join(f"iter_{total_rounds}.png")
        # cv2.imwrite(save_path, img_iter)
    print(f"迭代器件检测结束 | 总轮次: {total_rounds}, 第二轮后累计新增器件数量: {total_new_comps}")

    # 两阶段检测：对检出的 ic（box）裁剪放大再检一次 pin_tip（含 NMS 去重），补充/细化 IC 引脚
    # if vision_cfg.get("box_pin_crop", True) and comps:
    #     extra_pins = _detect_ic_pins_crop(img, comps, models.get("yolo_weights") or "",
    #                                       conf=float(vision_cfg.get("box_pin_conf", 0.15)))
    #     if extra_pins:
    #         # 合并策略：全图 pin 优先保留；裁剪 pin 只在 8px 内没有现有点时补充
    #         # （裁剪高置信度点若挤掉全图点，反而会让 tip 漂移）
    #         merged = list(pins)
    #         for pp in extra_pins:
    #             if any((pp.xy[0] - q.xy[0]) ** 2 + (pp.xy[1] - q.xy[1]) ** 2 < 64 for q in merged):
    #                 continue
    #             merged.append(pp)
    #         pins = merged

    if not comps:
        # No detector: rule-based component fallback (weak, low score expected).
        work, scale = _maybe_resize(img, max_side)
        comps = _rule_components(work, max_components=max_components)
        if scale < 1.0:
            for box in comps:
                box.bbox = [v / scale for v in box.bbox]

    # 文字识别完全由整图 PaddleOCR 完成：
    #   - 不读 ocr_cache（每次推理都重新识别）；
    #   - 不依赖 YOLO 的 text 框（YOLO 不预测 text 类）；
    #   - 对**原分辨率整图**做 det+rec（paddle 内部自行缩放，det_limit_side_len 控制），
    #     不再额外预缩放（预缩放会把字缩太小，大图识别率暴跌）。
    texts: list[TextBox] = []
    if ocr_enabled:
        texts = _full_image_ocr(image_path, lang=lang)

    skel, segs = None, None
    # wired = _try_unet_wires(img, models.get("unet_weights") or "", max_side, max_segments)
    # unet_points: list[tuple[float, float]] = []
    # if wired is None:
    #     skel, segs = _skeleton_wires(img, max_side=max_side, max_segments=max_segments)
    # else:
    #     skel, segs, unet_points = wired

    # UNet 连接点并入端点列表（网络重建时作为 wire_endpoint 节点参与）
    all_endpoints = list(endpoints)
    # for (px, py) in unet_points:
    #     all_endpoints.append(PinBox(xy=(px, py), score=0.6))

    # 可视化（agents/ 批量跑时可用 visualize=False 跳过，不影响检测结果）
    if visualize:
        run_vision_visualize(image_path, img, comps, texts, pins, endpoints)
        run_gt_visualize(image_path, img)

    pins, nets = extract_nets(img, comps, texts)
    save_path = draw_pcb_pin_net(img, pins, nets, case_name=Path(image_path).parent.name)
    print(f"可视化已保存：{save_path}")
    # out = extract_wire_nets(img, image_path, pins, comps, all_endpoints)
    # if visualize:
    #     run_wire_visualize(out, img, image_path)

    # return VisionOut(height=h, width=w, components=comps, texts=texts, pins=pins, endpoints=endpoints, skeleton=skel, wire_segments=segs, wire_nets=out or {}, img=img, image_path=image_path)
