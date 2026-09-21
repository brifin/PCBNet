#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build OCR cache for the 10gt test images (runs in the paddle OCR venv)."""

import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, '/data/luoxuan/PCBNet')

from paddleocr import PaddleOCR
from pcbnet.config import load_config, project_root
from pcbnet.ocr_cache import ocr_cache_file, save_json

cfg = load_config('/data/luoxuan/PCBNet/configs/default.yaml')
out_root = project_root() / cfg.get('output_root', 'outputs1')
test_root = Path(cfg['test_root'])
max_side = int((cfg.get('vision') or {}).get('max_side', 1600))
lang = str((cfg.get('ocr') or {}).get('lang') or 'en')

ocr = PaddleOCR(lang=lang, device='gpu:0', use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False)


def read_image(p):
    raw = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def parse(result):
    out = []
    if result is None:
        return out
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue
        if isinstance(page, dict) or hasattr(page, 'get'):
            rt = list(page.get('rec_texts') or page.get('rec_text') or [])
            rs = list(page.get('rec_scores') or page.get('rec_score') or [1.0]*len(rt))
            rp = list(page.get('rec_polys') or page.get('dt_polys') or page.get('rec_boxes') or [])
            for i, txt in enumerate(rt):
                poly = rp[i] if i < len(rp) else [[0,0],[1,0],[1,1],[0,1]]
                if hasattr(poly, 'tolist'):
                    poly = poly.tolist()
                if isinstance(poly, (list, tuple)) and len(poly) == 4 and not isinstance(poly[0], (list, tuple)):
                    x1,y1,x2,y2 = [float(v) for v in poly]
                    bbox = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
                else:
                    xs = [float(p[0]) for p in poly]; ys = [float(p[1]) for p in poly]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                conf = float(rs[i]) if i < len(rs) else 1.0
                if str(txt).strip():
                    out.append((bbox, str(txt).strip(), conf))
            continue
        if not isinstance(page, list):
            continue
        for item in page:
            if not item:
                continue
            box, payload = item[0], item[1]
            txt, conf = payload[0], payload[1]
            xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
            if str(txt).strip():
                out.append(([min(xs),min(ys),max(xs),max(ys)], str(txt).strip(), float(conf)))
    return out


imgs = sorted(p for p in test_root.rglob('*.png') if p.parent.name != test_root.name)
n = 0
for img_path in imgs:
    img = read_image(img_path)
    h, w = img.shape[:2]
    oi = img
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / max(h, w)
        oi = cv2.resize(img, (max(1,int(w*scale)), max(1,int(h*scale))), interpolation=cv2.INTER_AREA)
    result = ocr.predict(oi)
    texts = parse(result)
    payload = []
    for bbox, content, score in texts:
        x1,y1,x2,y2 = bbox
        if oi.shape[0] != h or oi.shape[1] != w:
            sx, sy = w/oi.shape[1], h/oi.shape[0]
            x1,y1,x2,y2 = x1*sx, y1*sy, x2*sx, y2*sy
        payload.append({'bbox':[float(x1),float(y1),float(x2),float(y2)], 'content':content, 'score':float(score)})
    save_json(ocr_cache_file(out_root, img_path), payload)
    n += 1
    print(f'[ocr] {img_path.parent.name} -> {len(payload)} texts')
print(f'done: {n} ocr cache files')
