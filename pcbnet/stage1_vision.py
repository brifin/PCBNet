"""Stage 1: component/text detection, OCR, wire skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from .visual import run_vision_visualize, run_gt_visualize, run_wire_visualize, draw_pcb_pin_net
from .topo_graph import extract_wire_nets
from .skele_wire import extract_nets as extract_skele_nets
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
    angle: int = 0          # 0 = 正向；90/270 = 旋转补充识别的候选
    source: str = "main"    # main / focus / rescue


@dataclass
class PinBox:
    xy: tuple[float, float]  # tip point in image coords
    score: float = 1.0
    number: str | None = None
    name: str | None = None
    # skele_wire 专属：归属元件在 vision.components 中的下标 / 元件内引脚序号
    owner: int | None = None
    idx: int | None = None


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
    # skele_wire.extract_nets 的原始输出（comp_<i>.pin_<j> 引用），stage2 用它做精确映射
    skele_pins: dict = field(default_factory=dict)
    skele_nets: dict = field(default_factory=dict)
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
# OCR 运行时与具体虚拟环境解耦：统一安装到通用缓存目录，运行时只设置 PYTHONPATH。
_OCR_PYTHONPATH: str | None = None
_OCR_RUNTIME_DIR = Path(
    os.environ.get("PCBNET_OCR_RUNTIME")
    or (Path.home() / ".cache" / "pcbnet" / "ocr37")
)


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


_YOLO_MODELS: dict[str, Any] = {}


def _get_yolo(weights: str):
    """按权重路径缓存 YOLO 实例，避免每张图重复加载 100+ MB 权重。"""
    from ultralytics import YOLO

    model = _YOLO_MODELS.get(weights)
    if model is None:
        model = YOLO(weights)
        _YOLO_MODELS[weights] = model
    return model


def _try_yolo(img: np.ndarray, weights: str, conf: float = 0.4,
              imgsz: int | None = None, iou: float = 0.7,
              max_det: int = 1000) -> list[DetBox]:
    if not weights:
        return []

    model = _get_yolo(weights)
    kw = dict(conf=conf, iou=iou, max_det=max_det, verbose=False)
    if imgsz:
        kw["imgsz"] = int(imgsz)
    try:
        res = model.predict(img, device=_resolve_yolo_device(), **kw)[0]
    except Exception as exc:  # noqa: BLE001 设备不可用时回退 CPU 再试
        print(f"[vision] 推理设备失败({exc})，回退 CPU", flush=True)
        global _YOLO_DEVICE
        _YOLO_DEVICE = "cpu"
        res = model.predict(img, device="cpu", **kw)[0]
    boxes = []
    names = res.names
    for b in res.boxes:
        xyxy = b.xyxy[0].tolist()
        cls = int(b.cls[0])
        boxes.append(DetBox(bbox=xyxy, label=str(names.get(cls, cls)), score=float(b.conf[0])))
    return boxes


def _try_yolo_multiscale(img: np.ndarray, weights: str, conf: float,
                         imgsz_list, merge_iou: float, iou: float = 0.7,
                         max_det: int = 1000) -> list[DetBox]:
    if not weights:
        return []
    if isinstance(imgsz_list, (int, float)):
        imgsz_list = [int(imgsz_list)]
    scales = [int(v) for v in (imgsz_list or []) if int(v) > 0]
    if not scales:
        return _try_yolo(img, weights, conf=conf, iou=iou, max_det=max_det)
    dets: list[DetBox] = []
    for imgsz in scales:
        dets.extend(_try_yolo(img, weights, conf=conf, imgsz=imgsz, iou=iou, max_det=max_det))
    return _nms_components(dets, iou_thr=merge_iou)


def _try_yolo_main(img: np.ndarray, models: dict, vision_cfg: dict) -> list[DetBox]:
    """单主模型多尺度推理（旧专家并集已移除）。"""
    main_w = models.get("yolo_weights") or ""
    main_conf = float(vision_cfg.get("yolo_conf", 0.25))
    scale_iou = float(vision_cfg.get("yolo_iou", 0.7))
    merge_iou = float(vision_cfg.get("yolo_scale_nms_iou", 0.55))
    max_det = int(vision_cfg.get("yolo_max_det", 1000))
    scales = vision_cfg.get("yolo_imgsz") or [1280]
    return _try_yolo_multiscale(
        img, main_w, main_conf, scales, merge_iou, iou=scale_iou, max_det=max_det
    )


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


def _ocr_subprocess_env(pythonpath: str | None) -> dict[str, str]:
    """OCR 子进程环境：PYTHONPATH 指向 3.7 运行时；容器里补 libgomp 搜索路径。"""
    env = dict(os.environ)
    # 评测平台会把 stderr 当 ingestion 日志：OCR 子进程里的 paddle/ccache 警告
    # 不是错误，统一静音；真正的异常仍通过子进程返回码/异常抛出。
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("GLOG_minloglevel", "2")
    env.setdefault("FLAGS_logtostderr", "0")
    if pythonpath:
        # worker 会把这个变量对应的目录显式插到 sys.path[0]；PYTHONPATH 只是兜底，
        # 避免镜像里的 sitecustomize/.pth 把旧版 paddleocr 抢到前面。
        env["PCBOCR_RUNTIME_PATH"] = pythonpath
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pythonpath + (":" + old if old else "")
    # CPU OCR 不需要 GPU；隐藏 GPU 可避免驱动/运行库探测带来的偶发导入失败。
    if not str(env.get("PCBOCR_DEVICE", "")).lower().startswith("gpu"):
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    project = Path(__file__).resolve().parent.parent
    libs_dir = project / "libs"
    if libs_dir.is_dir():
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(libs_dir) + (":" + old_ld if old_ld else "")
    return env


def _probe_ocr_version(pythonpath: str | None) -> str:
    """用当前解释器探测 paddleocr 版本；失败返回空串并打印真实错误。"""
    code = "import sys"
    if pythonpath:
        # 显式插到最前面：PYTHONPATH 可能被镜像的 .pth/sitecustomize 垫后。
        code += f"; sys.path.insert(0, {str(pythonpath)!r})"
    code += (
        "; import paddleocr; "
        "print('PCBOCR_VERSION=' + str(getattr(paddleocr, '__version__', '')))"
    )
    try:
        env = _ocr_subprocess_env(pythonpath)
        if pythonpath:
            # 探测某个候选目录时不要继承外部 PYTHONPATH，否则别的 3.7 会把
            # 一个不完整的候选目录“伪装”成可用。
            env["PYTHONPATH"] = pythonpath
        out = subprocess.run(
            ([sys.executable, "-S"] if pythonpath else [sys.executable]) + ["-c", code],
            env=env,
            capture_output=True, text=True, errors="replace", timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ocr] 探测 PaddleOCR 失败：{exc}", flush=True)
        return ""
    if out.returncode != 0:
        err = [x for x in (out.stderr or "").strip().splitlines() if x.strip()]
        print(
            f"[ocr] 探测 PaddleOCR 失败（rc={out.returncode}）："
            + " | ".join(err[-6:]),
            flush=True,
        )
        return ""
    for line in reversed((out.stdout or "").splitlines()):
        if line.strip().startswith("PCBOCR_VERSION="):
            return line.strip().split("=", 1)[1].strip()
    return ""


def _pylibs_dir() -> Path | None:
    """随包 OCR 运行时目录（提交镜像里构建阶段已装好，无需再探测/联网安装）。"""
    d = Path(__file__).resolve().parent.parent / "pylibs"
    return d if (d / "paddleocr").is_dir() else None


def _ensure_ocr37() -> str:
    """确保存在 PaddleOCR 3.7 运行时，返回需要加到 PYTHONPATH 的目录。

    不依赖任何特定虚拟环境：
      1. 容器/打包目录里有 pylibs（已装 3.7）→ 直接用，不探测、不联网；
      2. 通用缓存目录 ~/.cache/pcbnet/ocr37 已是 3.7 → 直接用；
      3. 当前解释器本身已是 3.7 → 不加 PYTHONPATH；
      4. 都没有 → pip 安装到通用缓存目录（帮助本地用户自动装好）。
    """
    global _OCR_PYTHONPATH
    if _OCR_PYTHONPATH is not None:
        return _OCR_PYTHONPATH

    bundled = _pylibs_dir()
    if bundled is not None:
        _OCR_PYTHONPATH = str(bundled)
        print(f"[ocr] 使用随包 PaddleOCR 3.7：{bundled}", flush=True)
        return _OCR_PYTHONPATH

    ver = _probe_ocr_version(str(_OCR_RUNTIME_DIR))
    if ver.startswith("3.7"):
        _OCR_PYTHONPATH = str(_OCR_RUNTIME_DIR)
        print(f"[ocr] PaddleOCR {ver}：{_OCR_RUNTIME_DIR}", flush=True)
        return _OCR_PYTHONPATH

    ver = _probe_ocr_version(None)
    if ver.startswith("3.7"):
        _OCR_PYTHONPATH = ""
        print(f"[ocr] PaddleOCR {ver}：当前环境", flush=True)
        return _OCR_PYTHONPATH

    _OCR_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[ocr] 未检测到 PaddleOCR 3.7（当前 {ver or '未安装'}），"
        f"正在安装到 {_OCR_RUNTIME_DIR}（首次约 1~2 分钟）…",
        flush=True,
    )
    base = [
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--no-input", "--disable-pip-version-check",
        "--target", str(_OCR_RUNTIME_DIR),
        "paddlepaddle==3.3.0", "paddleocr==3.7.0", "paddlex==3.7.2",
    ]
    for extra in (["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"], []):
        try:
            proc = subprocess.run(base + extra, timeout=1800)
            if proc.returncode == 0:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[ocr] pip 安装失败（{exc}），尝试备用源…", flush=True)

    ver = _probe_ocr_version(str(_OCR_RUNTIME_DIR))
    if not ver.startswith("3.7"):
        raise RuntimeError(
            "PaddleOCR 3.7 安装失败。请手动执行：\n"
            f"  {sys.executable} -m pip install --target {_OCR_RUNTIME_DIR} "
            "paddlepaddle==3.3.0 paddleocr==3.7.0 paddlex==3.7.2"
        )
    _OCR_PYTHONPATH = str(_OCR_RUNTIME_DIR)
    print(f"[ocr] PaddleOCR {ver} 安装完成：{_OCR_RUNTIME_DIR}", flush=True)
    return _OCR_PYTHONPATH


def _ensure_ocr_models() -> None:
    """PP-OCRv6 模型：优先复用已缓存；没有时从随包 ocr_models 复制。"""
    cache = Path(
        os.environ.get("PADDLE_PDX_CACHE_HOME")
        or (Path.home() / ".paddlex")
    )
    need = ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec")
    model_dir = cache / "official_models"
    if all((model_dir / n).is_dir() for n in need):
        return

    bundled = (
        Path(__file__).resolve().parent.parent
        / "submission" / "ocr_models" / ".paddlex" / "official_models"
    )
    if not bundled.is_dir():
        # 没有随包模型时交给 PaddleX 在首次推理时下载。
        return
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in need:
            src = bundled / name
            dst = model_dir / name
            if src.is_dir() and not dst.is_dir():
                shutil.copytree(src, dst)
        print(f"[ocr] 已准备 PP-OCRv6 模型：{model_dir}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[ocr] 复制 OCR 模型失败（将由 PaddleX 尝试下载）：{exc}", flush=True)


def _full_image_ocr(
    image_path: str | Path,
    lang: str = "en",
    focus_boxes: list[dict] | None = None,
) -> list[TextBox]:

    """每次推理都做一次**全图 OCR**（不读缓存）。

    focus_boxes 为需要局部补识别的元件框（box / gnd / v / net_label），worker
    会对这些区域放大后重新 det+rec，并对竖排文字做旋转识别，补回小字/竖排内容。

    torch 已加载时（主推理进程），paddlepaddle 与 torch 的 CUDA/运行库可能冲突，
    改用独立子进程跑 OCR（pcbnet/_ocr_worker.py）；OCR 固定使用 PaddleOCR 3.7，
    当前环境没有时自动安装到通用缓存目录并仅对 OCR 子进程设置 PYTHONPATH。
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
        # 不管当前是什么环境，都先把 PaddleOCR 3.7 运行时准备好；
        # 没有就自动安装到通用缓存目录，并只通过 PYTHONPATH 注入 OCR 子进程。
        pythonpath = _ensure_ocr37()
        _ensure_ocr_models()
        interpreter = sys.executable
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            out_path = fh.name
        focus_path = ""
        if focus_boxes:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as fh:
                json.dump(focus_boxes, fh, ensure_ascii=False)
                focus_path = fh.name
        try:
            env = _ocr_subprocess_env(pythonpath)
            env.setdefault("PCBOCR_LANG", lang)
            # 自动安装的是 paddlepaddle 3.3.0 CPU；需要 GPU 时可显式设 PCBOCR_DEVICE=gpu:0。
            env.setdefault("PCBOCR_DEVICE", "cpu")
            # 有显式 OCR 运行库时用 -S 隔离：只加载标准库 + pylibs/缓存目录，
            # 不导入镜像 site-packages，彻底避开镜像里旧版 paddleocr/冲突包。
            cmd = [interpreter]
            if pythonpath:
                cmd.append("-S")
            cmd += [str(worker), str(image_path), out_path]
            if focus_path:
                cmd.append(focus_path)
            subprocess.run(cmd, check=True, env=env, timeout=1800)
            data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        finally:
            for p in (out_path, focus_path):
                if not p:
                    continue
                try:
                    Path(p).unlink()
                except OSError:
                    pass
        return [
            TextBox(
                bbox=list(d["bbox"]),
                content=d["content"],
                score=float(d["score"]),
                angle=int(d.get("angle") or 0),
                source=str(d.get("src") or "main"),
            )
            for d in data
        ]
    img = _read_image(image_path)
    return _ocr_texts(img, enabled=True, lang=lang)


def _maybe_resize(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


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


def _adapt_skele_output(
    pins_dict: dict, nets_dict: dict
) -> tuple[list[PinBox], dict]:
    """把 skele_wire.extract_nets 的输出适配成 VisionOut 需要的结构。

    skele_wire 输出：
        pins_dict = {"comp_<i>": {"pin_<j>": {"point": {"x","y"}}}}
        nets_dict = {"net_<k>": {"hyperGraph": "(comp_1.pin_0,...)",
                                 "edges": {"edge_0": [{"x","y"}, {"x","y"}]}}}
    这里生成：
      - pins: PinBox 列表（带 owner=元件下标 / idx=元件内引脚序号），供 stage2
        生成 pins 并在构建 nets 时按 ref 精确映射；
      - wire_nets: 兼容旧 topo_graph 的坐标格式（all_pins + edges），作为兜底。
    """
    pins: list[PinBox] = []
    pos: dict[str, tuple[float, float]] = {}
    for ckey, pin_map in (pins_dict or {}).items():
        try:
            cid = int(str(ckey).rsplit("_", 1)[1])
        except Exception:  # noqa: BLE001
            continue
        for pkey, info in (pin_map or {}).items():
            pt = (info or {}).get("point") or {}
            if "x" not in pt or "y" not in pt:
                continue
            xy = (float(pt["x"]), float(pt["y"]))
            try:
                pidx = int(str(pkey).rsplit("_", 1)[1])
            except Exception:  # noqa: BLE001
                pidx = None
            pins.append(PinBox(xy=xy, score=1.0, owner=cid, idx=pidx))
            pos[f"{ckey}.{pkey}"] = xy

    wire_nets: dict = {}
    for net_key, net in (nets_dict or {}).items():
        net = net or {}
        refs = [r.strip() for r in str(net.get("hyperGraph") or "").strip("()").split(",") if r.strip()]
        all_pins = []
        for ref in refs:
            xy = pos.get(ref)
            if xy is not None:
                all_pins.append({"x": xy[0], "y": xy[1], "type": "pin", "ref": ref})
        edges: dict = {}
        for ek, seg in (net.get("edges") or {}).items():
            pts = [{"x": float(p["x"]), "y": float(p["y"])} for p in seg if "x" in p and "y" in p]
            if pts:
                edges[str(ek)] = pts
        if all_pins or edges:
            wire_nets[str(net_key)] = {"all_pins": all_pins, "edges": edges}
    return pins, wire_nets


def extract_vision(
    image_path: str | Path, cfg: dict[str, Any] | None = None, visualize: bool = True
) -> VisionOut:
    cfg = cfg or {}
    image_path = Path(image_path)
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

    # 元件检测：主模型（32 类全部类别）+ 低类专家模型（power/ground/
    # network_input/led 并集）。新模型不预测 pin_tip / wire_endpoint / text，
    # 引脚与网络完全交给 skele_wire.extract_nets。
    dets = _try_yolo_main(img, models, vision_cfg)
    comps, text_boxes, pins, endpoints = _split_detections(dets)
    comps = _clean_components(comps, text_boxes)
    comps = _nms_components(comps, iou_thr=float(vision_cfg.get("comp_nms_iou", 0.5)))
    # 新方案不再使用 YOLO 的 pin_tip / wire_endpoint，也不做 box 引脚裁剪补检
    pins, endpoints = [], []

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
    #   - 整图 det+rec；再对 box 内部、缺标签的 gnd/v/net 元件做局部放大识别，
    #     并对竖排文本框旋转补识别（小字/竖排 GND、IC pinname/pin_number）。
    texts: list[TextBox] = []
    if ocr_enabled:
        focus = [
            {"bbox": list(c.bbox), "label": c.label}
            for c in comps
            if c.label in {"ic", "gnd", "vcc", "net_label",
                           "ground", "power", "network_input", "network_output",
                           "network_short", "network_bidirection"}
        ]
        texts = _full_image_ocr(image_path, lang=lang, focus_boxes=focus)

    wired = _try_unet_wires(img, models.get("unet_weights") or "", max_side, max_segments)
    unet_points: list[tuple[float, float]] = []
    if wired is None:
        skel, segs = _skeleton_wires(img, max_side=max_side, max_segments=max_segments)
    else:
        skel, segs, unet_points = wired

    # 可视化（agents/ 批量跑时可用 visualize=False 跳过，不影响检测结果）
    if visualize:
        run_vision_visualize(image_path, img, comps, texts, skel, pins, endpoints)
        run_gt_visualize(image_path, img)

    # pin / net 主路径：skele_wire.extract_nets（fix net extraction）
    skele_pins_raw: dict = {}
    skele_nets_raw: dict = {}
    wire_nets: dict = {}
    skele_pins: list[PinBox] = []
    try:
        skele_pins_raw, skele_nets_raw = extract_skele_nets(
            img, comps, texts, str(image_path), visualize=visualize
        )
        skele_pins, wire_nets = _adapt_skele_output(skele_pins_raw, skele_nets_raw)
    except Exception as exc:  # noqa: BLE001 异常兜底：退回旧的 topo_graph
        print(f"[vision] skele_wire.extract_nets 失败({type(exc).__name__}: {exc})，退回 topo_graph", flush=True)
        import traceback

        traceback.print_exc(limit=5)
        all_endpoints = list(endpoints) + [PinBox(xy=(px, py), score=0.6) for (px, py) in unet_points]
        try:
            wire_nets = extract_wire_nets(img, image_path, pins, comps, all_endpoints, visualize=False) or {}
        except Exception as exc2:  # noqa: BLE001
            print(f"[vision] topo_graph 兜底也失败({exc2})，nets 置空", flush=True)
            wire_nets = {}

    if visualize and wire_nets:
        run_wire_visualize(wire_nets, img, image_path)
        if skele_pins_raw and skele_nets_raw:
            try:
                draw_pcb_pin_net(img, skele_pins_raw, skele_nets_raw,
                                 Path(image_path).parent.name, output_root="outputs/viz")
            except Exception as exc:  # noqa: BLE001
                print(f"[viz] pin_net_visual 生成失败: {exc}")

    return VisionOut(
        height=h, width=w, components=comps, texts=texts,
        pins=skele_pins, endpoints=endpoints,
        skeleton=skel, wire_segments=segs, wire_nets=wire_nets,
        skele_pins=skele_pins_raw, skele_nets=skele_nets_raw,
        img=img, image_path=image_path,
    )
