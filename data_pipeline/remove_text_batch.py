"""对 base_comps 中 amp_3pin ~ l 已提取的图片批量去文字。

逻辑与 extract_single.remove_text 一致（含面积上限，避免把元件符号
误检为大文字而整体擦除）：PaddleOCR 检测文字框（置信度 >= 0.6 且面积
<= 整图 60%），框内 alpha 置 0，覆盖保存原 png。

用法：
  .venv_ocr/bin/python data_pipeline/remove_text_batch.py            # 就地处理
  .venv_ocr/bin/python data_pipeline/remove_text_batch.py --from-backup  # 先从备份恢复再处理
"""

import argparse
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE = "/data/luoxuan/PCBNet/data/base_comps"
BACKUP = "/tmp/base_comps_before_text"
TYPE_DIRS = [
    "amp_3pin", "amp_5pin", "ant", "battary", "bjt", "bjt_npn", "bjt_pnp",
    "block", "buzzer", "c", "circle_header", "crystal", "crystal_2pin",
    "crystal_3pin", "crystal_4pin", "d", "dc", "esd", "fuse", "gnd", "jumper", "l",
]
CONF_THR = 0.6
MAX_AREA_RATIO = 0.6
PAD = 1


def _get_ocr():
    from paddleocr import PaddleOCR
    try:
        return PaddleOCR(lang="en", device="gpu:0",
                         use_doc_orientation_classify=False,
                         use_doc_unwarping=False, use_textline_orientation=False)
    except (TypeError, ValueError):
        return PaddleOCR(lang="en", use_gpu=True, use_angle_cls=True, show_log=False)


def _ocr_boxes(ocr, arr):
    """兼容 paddleocr 3.x(predict) / 2.x(ocr)。返回 [(poly, score)]。"""
    pairs = []
    if hasattr(ocr, "predict"):
        try:
            r = ocr.predict(arr)
            page = r[0] if r else {}
            polys = list(page.get("rec_polys") or [])
            scores = list(page.get("rec_scores") or [])
            return [(p, s) for p, s in zip(polys, scores) if s is not None]
        except Exception:
            pass
    try:
        for page in (ocr.ocr(arr, cls=True) or []):
            if not page:
                continue
            for item in page:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    pairs.append((item[0], item[2]))
    except Exception:
        pass
    return pairs


def remove_text(ocr, img: Image.Image) -> tuple[Image.Image, int]:
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    arr = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)
    boxes = _ocr_boxes(ocr, arr)
    if not boxes:
        return img, 0
    out = np.asarray(img).copy()
    h, w = out.shape[:2]
    total = float(h * w) or 1.0
    n = 0
    for poly, score in boxes:
        try:
            if float(score) < CONF_THR:
                continue
        except (TypeError, ValueError):
            continue
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        x1 = max(0, int(min(xs)) - PAD)
        y1 = max(0, int(min(ys)) - PAD)
        x2 = min(w, int(max(xs)) + 1 + PAD)
        y2 = min(h, int(max(ys)) + 1 + PAD)
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) > MAX_AREA_RATIO * total:
            continue  # 过大框 = 元件符号误检，跳过
        out[y1:y2, x1:x2, 3] = 0
        n += 1
    return Image.fromarray(out), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-backup", action="store_true",
                    help="先从 /tmp/base_comps_before_text 恢复原始图再处理")
    args = ap.parse_args()

    if args.from_backup and os.path.isdir(BACKUP):
        for t in TYPE_DIRS:
            bd, cd = os.path.join(BACKUP, t), os.path.join(BASE, t)
            if not os.path.isdir(bd):
                continue
            os.makedirs(cd, exist_ok=True)
            for f in os.listdir(bd):
                if f.endswith(".png") and "_vis" not in f:
                    shutil.copy(os.path.join(bd, f), os.path.join(cd, f))
        print("已从备份恢复原始图")

    ocr = _get_ocr()
    files = []
    for t in TYPE_DIRS:
        d = os.path.join(BASE, t)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".png") and "_vis" not in f:
                files.append(os.path.join(d, f))
    print(f"待处理: {len(files)} 张")

    total_erased = 0
    for p in tqdm(files, desc="去文字"):
        img = Image.open(p).convert("RGBA")
        out, n = remove_text(ocr, img)
        if n:
            out.save(p)
            total_erased += n
    print(f"完成: {len(files)} 张, 共擦除 {total_erased} 处文字")


if __name__ == "__main__":
    main()
