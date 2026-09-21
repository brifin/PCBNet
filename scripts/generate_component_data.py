#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate enlarged single-component crops for low-mAP (hard) component classes.

For each original case, pick one component per "hard" type (whose coarse class has
mAP < 70%), crop the region (component bbox + pin tips + margin), then place the
resized crop onto a NEW white image of the SAME size as the original, so the
component occupies ~50% of the new image. Coordinates are recomputed from the scale
factor and placement. JSON holds only that component and its pins (no nets).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.label_export import load_coarse_map  # noqa: E402

TRAIN_DIR = ROOT / "data" / "Dataset200" / "200_train_cases"
MARGIN = 10
TOOL = "crop"
TARGET_FRAC = 0.5     # 元件最大边约占新图 50%
MAX_SCALE = 20.0      # 防止极小符号被过度放大

# 逐类别 mAP < 70% 的 coarse 类（来自训练结果）
BAD_COARSE = {"capacitor", "transistor", "gnd", "vcc", "connector", "crystal",
              "switch", "testpoint", "other"}


def coarse_of(race: str, coarse_map: dict[str, str]) -> str:
    return coarse_map.get(race) or "other"


def read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def main() -> int:
    coarse_map = load_coarse_map()
    case_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    next_id = max(int(d.name) for d in case_dirs) + 1
    print(f"existing cases: {len(case_dirs)}, next id starts at {next_id}")

    n_new = 0
    for d in sorted(case_dirs, key=lambda x: int(x.name)):
        json_path = next(d.glob("*_target*.json"))
        img_path = next(p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        data = json.loads(json_path.read_text(encoding="utf-8"))
        img = read_image(img_path)
        H, W = img.shape[:2]

        # one component per distinct HARD type (mAP < 70%)
        first_of_type: dict[str, str] = {}
        for key, comp in (data.get("components") or {}).items():
            t = comp.get("type") or "other"
            if coarse_of(t, coarse_map) in BAD_COARSE:
                first_of_type.setdefault(t, key)

        for ctype, key in first_of_type.items():
            comp = data["components"][key]
            pins = (data.get("pins") or {}).get(key) or {}
            bx = comp.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])

            xs = [x1, x2]
            ys = [y1, y2]
            for pk, pin in pins.items():
                pt = pin.get("point") or {}
                xs.append(float(pt.get("x", 0)))
                ys.append(float(pt.get("y", 0)))

            # 裁剪区域（contest 坐标，y 向上），clamp 到图内
            cx1 = max(0.0, min(xs) - MARGIN)
            cy1 = max(0.0, min(ys) - MARGIN)
            cx2 = min(float(W), max(xs) + MARGIN)
            cy2 = min(float(H), max(ys) + MARGIN)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            # 裁剪（图像坐标 y 向下：行 [H-cy2, H-cy1]）
            crop = img[int(H - cy2):int(H - cy1), int(cx1):int(cx2)]
            if crop.size == 0:
                continue

            # 缩放：元件最大边占新图 ~50%
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            s = max(1.0, TARGET_FRAC * min(W / bw, H / bh))
            # 保证缩放后的裁剪区域仍落在新图内
            s = min(s, (W - 4) / (cx2 - cx1), (H - 4) / (cy2 - cy1))

            rcrop = cv2.resize(crop, (max(1, int(round(s * crop.shape[1]))),
                                      max(1, int(round(s * crop.shape[0])))),
                               interpolation=cv2.INTER_CUBIC)
            rcw, rch = rcrop.shape[1], rcrop.shape[0]

            # 白底新图（与原图同尺寸），居中放置
            new_img = np.full((H, W, 3), 255, np.uint8)
            px = max(0, (W - rcw) // 2)
            py = max(0, (H - rch) // 2)
            new_img[py:py + rch, px:px + rcw] = rcrop

            # 重算坐标：元件 bbox
            # crop 内（图像坐标）：left=x1-cx1, right=x2-cx1, top=cy2-y2, bottom=cy2-y1
            nl = px + s * (x1 - cx1)
            nr = px + s * (x2 - cx1)
            nt = py + s * (cy2 - y2)
            nb = py + s * (cy2 - y1)
            # 转 contest（y 向上）
            new_bbox = [nl, H - nb, nr, H - nt]

            new_comp = dict(comp)
            new_comp["bbox"] = new_bbox
            new_pins = {}
            for pk, pin in pins.items():
                pt = pin.get("point") or {}
                ix = float(pt.get("x", 0)) - cx1
                iy = cy2 - float(pt.get("y", 0))
                nix = px + s * ix
                niy = py + s * iy
                new_pins[pk] = {"pinname": pin.get("pinname"),
                                "point": {"x": nix, "y": H - niy}}

            new_data = {"components": {key: new_comp}, "pins": {key: new_pins}, "nets": {}}

            # 二值化：白底 + 黑线（统一彩色线条为黑色）
            gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            new_img = cv2.cvtColor(np.where(gray < 200, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            new_dir = TRAIN_DIR / f"{next_id:04d}"
            new_dir.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", new_img)
            if not ok:
                continue
            buf.tofile(str(new_dir / f"{next_id:04d}-{TOOL}.png"))
            (new_dir / f"{next_id:04d}-{TOOL}_target_named.json").write_text(
                json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            next_id += 1
            n_new += 1

    print(f"generated {n_new} new cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
