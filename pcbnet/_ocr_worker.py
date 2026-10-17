#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全图 OCR 子进程 worker。

主进程里 torch 与 paddlepaddle-gpu 的 CUDA 库会冲突（libcublas 符号错误），
所以整图 OCR 放到独立子进程执行：本脚本只 import paddleocr，对整图做 det+rec，
把结果写 JSON。不读任何缓存——每次调用都是全新识别。

用法：python _ocr_worker.py <图片路径> <输出json路径>
环境变量：PCBOCR_DEVICE（默认 cpu；3.x 用 gpu:0）、PCBOCR_DET_LIMIT（默认 2560）、
          PCBOCR_LANG（默认 en）。
兼容 paddleocr 2.x 与 3.x；提交容器内依赖装在 /submission/pylibs，这里自动追加 sys.path。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 提交容器：paddle 在 /submission/pylibs（本机该目录不存在，自动忽略）
_PYLIBS = Path(__file__).resolve().parent.parent / "pylibs"
if _PYLIBS.is_dir():
    sys.path.append(str(_PYLIBS))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    img_path, out_path = sys.argv[1], sys.argv[2]
    dev = os.environ.get("PCBOCR_DEVICE", "cpu").strip() or "cpu"
    det_limit = int(os.environ.get("PCBOCR_DET_LIMIT", "2560"))
    lang = os.environ.get("PCBOCR_LANG", "en")

    img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        json.dump([], open(out_path, "w", encoding="utf-8"))
        return 1
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

    ver = str(getattr(paddleocr, "__version__", ""))
    out: list[dict] = []
    if ver.startswith("3"):
        engine = paddleocr.PaddleOCR(
            lang=lang, device=dev, use_doc_orientation_classify=False,
            use_doc_unwarping=False, use_textline_orientation=False,
            det_limit_side_len=det_limit,
        )
        res = engine.predict(img)
        item = res[0]
        texts = list(item.get("rec_texts") or [])
        scores = list(item.get("rec_scores") or [])
        boxes = list(item.get("rec_polys", item.get("dt_polys", [])) or [])
        for b, t, s in zip(boxes, texts, scores):
            if not str(t or "").strip():
                continue
            pts = np.asarray(b).reshape(-1, 2)
            out.append({
                "bbox": [float(pts[:, 0].min()) * inv_scale, float(pts[:, 1].min()) * inv_scale,
                         float(pts[:, 0].max()) * inv_scale, float(pts[:, 1].max()) * inv_scale],
                "content": str(t),
                "score": float(s),
            })
    else:
        engine = paddleocr.PaddleOCR(
            lang=lang, device=dev, use_angle_cls=False, show_log=False,
            det_limit_side_len=det_limit,
        )
        res = engine.ocr(img, cls=True)
        for box, (t, s) in (res[0] or []):
            if not str(t or "").strip():
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            out.append({
                "bbox": [min(xs) * inv_scale, min(ys) * inv_scale, max(xs) * inv_scale, max(ys) * inv_scale],
                "content": str(t),
                "score": float(s),
            })
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[ocr-worker] {Path(img_path).name}: {len(out)} texts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
