#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全图 OCR 子进程 worker。

主进程里 torch 与 paddle 的运行库/CUDA 库可能冲突，所以整图 OCR 放到独立子进程：
本脚本只 import paddleocr，对整图做 det+rec，把结果写 JSON。不读任何缓存。

固定使用 PaddleOCR 3.7 / PaddleX 3.7.2（PP-OCRv6）。主进程会先确保 3.7 运行时，
并通过 PYTHONPATH 传给本进程；当前环境不是 3.7 会直接报错，不会静默使用旧版 OCR。

除整图识别外，还会做两类补充识别（尺寸都很小，耗时可接受）：
  1. 竖排文字：对主识别中"高瘦"的文本框，把 crop 旋转 90° 再识别一次，作为候选；
  2. 元件局部：对 box（IC）内部和附近没有文字的 gnd/v/net_label 元件 crop 放大后
     det+rec（含旋转），补回小字号/竖排的 pin_name、pin_number 和电源标签。

用法：python _ocr_worker.py <图片路径> <输出json路径> [focus_boxes.json]
环境变量：PCBOCR_DEVICE（默认 cpu；GPU 用 gpu:0）、PCBOCR_DET_LIMIT（默认 2560）、
          PCBOCR_LANG（默认 en）、PCBOCR_MAX_SIDE（默认 1600）。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path

# 评测平台会把 stderr 当作 ingestion 日志；Paddle/PaddleX 的 INFO/ccache 警告
# 不是错误，却会让日志看起来像失败。这里在 import paddle 之前统一静音。
warnings.filterwarnings("ignore", message=".*ccache.*")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_minloglevel", "2")
os.environ.setdefault("FLAGS_logtostderr", "0")
logging.getLogger().setLevel(logging.ERROR)

# 主进程通过 PYTHONPATH 传 OCR 3.7 目录；这里再显式插到 sys.path[0]。
# 某些镜像的 sitecustomize/.pth 会把旧版 paddleocr 插到 PYTHONPATH 之前，
# 只靠环境变量可能导错版本。
_runtime = os.environ.get("PCBOCR_RUNTIME_PATH")
if _runtime:
    sys.path.insert(0, _runtime)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _quiet_paddle_logs() -> None:
    """PaddleX 用自己的 colorlog logger，import 后再压到 ERROR。"""
    logging.getLogger().setLevel(logging.ERROR)
    for name in ("paddlex", "paddle", "paddleocr", "ppocr", "PPOCR"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import paddlex  # noqa: F401

        paddlex.utils.logging._logger.setLevel(logging.ERROR)
    except Exception:
        pass


def _norm(txt: str) -> str:
    return "".join(str(txt or "").lower().split())


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center(bb: list[float]) -> tuple[float, float]:
    return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _page_items(page, inv_scale: float, src: str, angle: int = 0) -> list[dict]:
    """把一次 paddle 结果转成原图坐标的文本项。"""
    out: list[dict] = []
    if page is None:
        return out
    texts = list(page.get("rec_texts") or [])
    scores = list(page.get("rec_scores") or [])
    boxes = list(page.get("rec_polys", page.get("dt_polys", [])) or [])
    for b, t, s in zip(boxes, texts, scores):
        t = str(t or "").strip()
        if not t:
            continue
        pts = np.asarray(b).reshape(-1, 2)
        out.append({
            "bbox": [float(pts[:, 0].min()) * inv_scale, float(pts[:, 1].min()) * inv_scale,
                     float(pts[:, 0].max()) * inv_scale, float(pts[:, 1].max()) * inv_scale],
            "content": t,
            "score": float(s),
            "angle": int(angle),
            "src": src,
        })
    return out


def _merge_items(items: list[dict]) -> list[dict]:
    """同一内容的重复框合并（保留高置信度）；不同内容（不同朝向）都保留为候选。"""
    out: list[dict] = []
    for it in items:
        bb, t = it["bbox"], _norm(it["content"])
        if not t:
            continue
        dup = False
        for i, old in enumerate(out):
            old_t = _norm(old["content"])
            overlap = _iou(bb, old["bbox"]) > 0.3 or (
                old["bbox"][0] <= _center(bb)[0] <= old["bbox"][2]
                and old["bbox"][1] <= _center(bb)[1] <= old["bbox"][3]
            )
            if old_t == t:
                if _iou(bb, old["bbox"]) > 0.45 or _dist(_center(bb), _center(old["bbox"])) < 5.0:
                    if it["score"] > old["score"]:
                        out[i] = it
                    dup = True
                    break
            if not overlap:
                continue
            # 局部 crop 可能只认出型号的一部分（'IP5306' 之于 'IP5306-4.35V'）：
            # 保留更完整的那个，避免残缺文字被当成 key/Name。
            if t in old_t and len(t) < len(old_t):
                dup = True
                break
            if old_t in t and len(t) > len(old_t) and _iou(bb, old["bbox"]) > 0.45:
                out[i] = it
                dup = True
                break
        if not dup:
            out.append(it)
    return out


def _map_rot_bbox(bb: list[float], crop_w: int, crop_h: int, clockwise: bool) -> list[float]:
    """旋转 crop 上的 bbox -> 未旋转 crop 坐标。"""
    x1, y1, x2, y2 = bb
    if clockwise:
        # (x, y) -> (crop_h-1-y, x)
        return [y1, crop_h - 1 - x2, y2, crop_h - 1 - x1]
    # counter-clockwise: (x, y) -> (y, crop_w-1-x)
    return [crop_w - 1 - y2, x1, crop_w - 1 - y1, x2]


def _crop_ocr(engine, img: np.ndarray, box: list[float], scale: float, pad: int,
              src: str, keep_rotated_tall_only: bool, rotations: bool = True) -> list[dict]:
    """crop 放大后 det+rec（正向 + 两个 90° 方向）。返回原图坐标文本项。"""
    h, w = img.shape[:2]
    x1 = max(0, int(round(box[0])) - pad)
    y1 = max(0, int(round(box[1])) - pad)
    x2 = min(w, int(round(box[2])) + pad)
    y2 = min(h, int(round(box[3])) + pad)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return []
    crop = img[y1:y2, x1:x2]
    if scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    out: list[dict] = []
    variants = [("up", 0), ("cw", 90), ("ccw", 270)] if rotations else [("up", 0)]
    for tag, angle in variants:
        if tag == "up":
            work = crop
        elif tag == "cw":
            work = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        else:
            work = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        try:
            res = engine.predict(work)
        except Exception:
            continue
        if not res:
            continue
        page = res[0]
        items = _page_items(page, 1.0, src=src, angle=angle)
        for it in items:
            bb = it["bbox"]
            # crop 内坐标 -> 原图
            if angle == 0:
                ob = [bb[0] / scale + x1, bb[1] / scale + y1,
                      bb[2] / scale + x1, bb[3] / scale + y1]
            else:
                # 映射用未旋转 crop 的宽高（work 的宽高已交换）
                rb = _map_rot_bbox(bb, crop.shape[1], crop.shape[0], angle == 90)
                ob = [rb[0] / scale + x1, rb[1] / scale + y1,
                      rb[2] / scale + x1, rb[3] / scale + y1]
            it["bbox"] = ob
            if angle != 0:
                bw, bh = ob[2] - ob[0], ob[3] - ob[1]
                if keep_rotated_tall_only and bh < 1.1 * bw:
                    continue
            out.append(it)
    return out


def _rescued_items(engine, img: np.ndarray, main_items: list[dict]) -> list[dict]:
    """高瘦文本：旋转 crop 再识别，作为同位置的候选内容（如竖排 GND）。"""
    inner = engine.paddlex_pipeline._pipeline
    rec = getattr(inner, "text_rec_model", None)
    if rec is None:
        return []
    h, w = img.shape[:2]
    cands: list[tuple[float, dict]] = []
    for it in main_items:
        x1, y1, x2, y2 = it["bbox"]
        bw, bh = x2 - x1, y2 - y1
        if bh < 1.2 * bw or bh < 14 or bw < 5:
            continue
        if len(str(it.get("content") or "")) > 16:
            continue
        cands.append((bh / max(1.0, bw), it))
    cands.sort(key=lambda x: -x[0])
    crops: list[np.ndarray] = []
    metas: list[tuple[dict, int]] = []
    for _, it in cands[:60]:
        x1, y1, x2, y2 = it["bbox"]
        pad = 3
        cx1 = max(0, int(round(x1)) - pad)
        cy1 = max(0, int(round(y1)) - pad)
        cx2 = min(w, int(round(x2)) + pad)
        cy2 = min(h, int(round(y2)) + pad)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        # 旋转后字体高度约为原宽度，放大 3 倍便于识别
        crop = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        for angle, rot in ((90, cv2.ROTATE_90_CLOCKWISE), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)):
            crops.append(cv2.rotate(crop, rot))
            metas.append((it, angle))
    if not crops:
        return []
    out: list[dict] = []
    try:
        results = list(rec.predict(crops))
    except Exception:
        return []
    for (it, angle), res in zip(metas, results):
        if not res:
            continue
        txt = str(res.get("rec_text") or "").strip()
        score = float(res.get("rec_score") or 0.0)
        if txt:
            x1, y1, x2, y2 = it["bbox"]
            out.append({"bbox": [x1, y1, x2, y2], "content": txt,
                        "score": score, "angle": angle, "src": "rescue"})
    return out


def _plausible_label(content: str, label: str) -> bool:
    """主识别在该符号附近已有"像标签"的文字时，不再重复 crop 识别。

    注意 'AD' 这类短噪声不算覆盖（很可能就是竖排 GND 被读错），需要局部重识别。
    """
    t = "".join(str(content or "").lower().split())
    if len(t) < 2 or not re.search(r"[a-z]", t):
        return False
    if label == "gnd":
        return "gnd" in t or len(t) >= 4
    if label == "vcc":
        return bool(re.search(r"(vcc|vdd|vss|vbat|vin|vout|vref|vbus|\d+(?:[.,]\d+)?v)", t)) or len(t) >= 3
    return True


def _focus_items(engine, img: np.ndarray, main_items: list[dict],
                 focus_boxes: list[dict]) -> list[dict]:
    """对 box 内部 / 缺文字的 gnd-v-net 符号做局部放大识别。"""
    out: list[dict] = []
    # IC 优先，其次符号
    focus_boxes = sorted(focus_boxes, key=lambda f: 0 if str(f.get("label")) == "ic" else 1)
    for fb in focus_boxes[:14]:
        bb = [float(v) for v in (fb.get("bbox") or [])]
        if len(bb) != 4:
            continue
        label = str(fb.get("label") or "")
        if label != "ic":
            bc = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            covered = any(
                _dist(bc, _center(t["bbox"])) < 65 and _plausible_label(t["content"], label)
                for t in main_items
            )
            if covered:
                continue
            out.extend(_crop_ocr(engine, img, bb, scale=2.0, pad=50, src="focus",
                                 keep_rotated_tall_only=True, rotations=True))
            continue
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        scale = 3.0 if max(bw, bh) < 120 else 2.0
        if max(bw, bh) * scale > 1400:
            scale = max(1.0, 1400.0 / max(bw, bh))
        # IC 内部文字（pinname/pin_number/型号）尽量在原分辨率上再识别一次；
        # 只有高瘦的 IC 才做旋转补识别（竖排引脚名的可能性大，横向框基本是水平文字）。
        out.extend(_crop_ocr(engine, img, bb, scale=scale, pad=2, src="focus",
                             keep_rotated_tall_only=True, rotations=(bh > 1.15 * bw)))
    return out


def main() -> int:
    img_path, out_path = sys.argv[1], sys.argv[2]
    focus_path = sys.argv[3] if len(sys.argv) > 3 else ""
    dev = os.environ.get("PCBOCR_DEVICE", "cpu").strip() or "cpu"
    det_limit = int(os.environ.get("PCBOCR_DET_LIMIT", "2560"))
    lang = os.environ.get("PCBOCR_LANG", "en")

    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        json.dump([], open(out_path, "w", encoding="utf-8"))
        return 1
    focus_boxes: list[dict] = []
    if focus_path:
        try:
            focus_boxes = json.loads(Path(focus_path).read_text(encoding="utf-8"))
        except Exception:
            focus_boxes = []

    # 与历史最优设置一致：先缩到 PCB_OCR_MAX_SIDE（默认 1600）再交给 paddle
    # （paddle 内部按 det_limit_side_len 再缩放），实测该组合的匹配分数最好
    max_side = int(os.environ.get("PCB_OCR_MAX_SIDE", "1600"))
    inv_scale = 1.0
    if max_side > 0 and max(img.shape[:2]) > max_side:
        s = max_side / float(max(img.shape[:2]))
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
        inv_scale = 1.0 / s

    import paddleocr

    _quiet_paddle_logs()
    ver = str(getattr(paddleocr, "__version__", ""))
    if not ver.startswith("3.7"):
        raise RuntimeError(
            f"需要 PaddleOCR 3.7，实际导入的是 {ver!r}。"
            "主流程会自动安装 3.7 到通用缓存目录；如果看到本错误，"
            "请检查 PYTHONPATH 是否把旧版 paddleocr 提前了。"
        )

    engine = paddleocr.PaddleOCR(
        lang=lang,
        device=dev,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=det_limit,
        # paddlepaddle 3.3 CPU + oneDNN 在 PP-OCRv6 的某些算子上报
        # ConvertPirAttribute2RuntimeAttribute；关闭 MKLDNN 后可正常推理。
        enable_mkldnn=False,
    )
    res = engine.predict(img)
    main_items = _page_items(res[0], inv_scale, src="main") if res else []

    # 局部识别改用 max 限制，避免小 crop 被 paddle 内部放大到 4000+ 导致极慢
    inner = engine.paddlex_pipeline._pipeline
    inner.text_det_limit_side_len = 960
    inner.text_det_limit_type = "max"
    inner.text_det_max_side_limit = 1280

    # 补充识别在**原图**上做（不受上面的 1600 预缩放影响）
    full = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    extra: list[dict] = []
    if full is not None:
        if main_items:
            extra.extend(_rescued_items(engine, full, main_items))
        if focus_boxes:
            extra.extend(_focus_items(engine, full, main_items, focus_boxes))

    out = _merge_items(main_items + extra)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    if os.environ.get("PCBOCR_VERBOSE") == "1":
        print(f"[ocr-worker] {Path(img_path).name}: {len(main_items)} main + {len(extra)} extra "
              f"-> {len(out)} texts (paddleocr {ver})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
