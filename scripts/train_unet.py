#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a small U-Net on exported wire masks. Uses train/val only."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root
from pcbnet.dataset import IMAGE_EXTS

try:
    import cv2
except ImportError:
    cv2 = None


def _list_ids(img_dir: Path) -> list[str]:
    return sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _read_bgr(path: Path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _read_gray(path: Path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)


def _pad_square(img: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / float(max(h, w))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    if img.ndim == 2:
        out = np.zeros((size, size), dtype=img.dtype)
        out[:nh, :nw] = small
    else:
        out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        out[:nh, :nw] = small
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", default="")
    parser.add_argument("--train-splits", default="train",
                        help="comma list of split folders to train on, e.g. train,val,test for all 64")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("未安装 torch。建议在 Python 3.10–3.12 环境从 PyTorch 官网安装 CPU/CUDA 包：")
        print("  https://pytorch.org/get-started/locally/")
        return 1
    if cv2 is None:
        print("需要 opencv-python")
        return 1

    from pcbnet.unet import bce_dice_loss, build_unet, dice_coef

    cfg = load_config(args.config)
    unet_cfg = ((cfg.get("train") or {}).get("unet") or {})
    root = Path(args.label_root) if args.label_root else project_root() / "outputs" / "labels"
    train_split_names = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    train_pairs = [(root / "images" / s, root / "masks" / s) for s in train_split_names]
    train_pairs = [(im, mk) for im, mk in train_pairs if im.exists() and mk.exists()]
    val_img = root / "images" / "val"
    val_mask = root / "masks" / "val"
    if not train_pairs:
        print("缺少 train 图像/mask，先运行: py scripts\\prepare_labels.py --splits train,val")
        return 1

    class WireSet(Dataset):
        def __init__(self, pairs: list[tuple[Path, Path]], size: int, limit: int = 0) -> None:
            self.size = size
            self.items: list[tuple[Path, Path, str]] = []
            for img_dir, mask_dir in pairs:
                for cid in _list_ids(img_dir):
                    if (mask_dir / f"{cid}.png").exists():
                        self.items.append((img_dir, mask_dir, cid))
            if limit:
                self.items = self.items[:limit]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            img_dir, mask_dir, cid = self.items[idx]
            img_path = next((img_dir / f"{cid}{ext}" for ext in IMAGE_EXTS if (img_dir / f"{cid}{ext}").exists()), None)
            if img_path is None:
                raise FileNotFoundError(cid)
            img = _read_bgr(img_path)
            mask = _read_gray(mask_dir / f"{cid}.png")
            if img is None or mask is None:
                raise FileNotFoundError(cid)
            img = _pad_square(img, self.size, False)[:, :, ::-1].astype(np.float32) / 255.0
            mask = (_pad_square(mask, self.size, True) > 0).astype(np.float32)
            x = torch.from_numpy(img.transpose(2, 0, 1))
            y = torch.from_numpy(mask[None])
            return x, y

    epochs = 1 if args.smoke else int(unet_cfg.get("epochs", args.epochs))
    size = 256 if args.smoke else int(unet_cfg.get("size", args.size))
    batch = 1 if args.smoke else int(unet_cfg.get("batch", args.batch))
    lr = float(unet_cfg.get("lr", args.lr))
    limit = 2 if args.smoke else 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_set = WireSet(train_pairs, size, limit)
    val_set = WireSet([(val_img, val_mask)], size, limit) if val_img.exists() and val_mask.exists() else train_set
    if len(train_set) == 0:
        print("train 集合为空")
        return 1
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    model = build_unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    out_dir = project_root() / "outputs" / "models" / "unet"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice, best_path = -1.0, out_dir / "best.pt"
    history = []
    print(f"[unet] train={len(train_set)} val={len(val_set)} epochs={epochs} size={size} device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                dices.append(float(dice_coef(model(x), y).item()))
        mean_loss = sum(losses) / max(len(losses), 1)
        mean_dice = sum(dices) / max(len(dices), 1)
        history.append({"epoch": epoch, "loss": mean_loss, "val_dice": mean_dice})
        print(f"  epoch {epoch}/{epochs} loss={mean_loss:.4f} val_dice={mean_dice:.4f}")
        ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "val_dice": mean_dice}
        torch.save(ckpt, out_dir / "last.pt")
        if mean_dice >= best_dice:
            best_dice = mean_dice
            torch.save(ckpt, best_path)
    pointer = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best": str(best_path),
        "val_dice": best_dice,
        "history": history,
        "device": device,
    }
    (out_dir / "last_run.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    print("训练完成。把下面路径填进 configs/default.yaml 的 models.unet_weights：")
    print(f"  {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
