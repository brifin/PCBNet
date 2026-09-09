"""Small U-Net for schematic wire segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_unet(in_ch: int = 3, out_ch: int = 1, base: int = 16):
    import torch
    from torch import nn

    class DoubleConv(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d1 = DoubleConv(in_ch, base)
            self.d2 = DoubleConv(base, base * 2)
            self.d3 = DoubleConv(base * 2, base * 4)
            self.d4 = DoubleConv(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.u3 = DoubleConv(base * 8 + base * 4, base * 4)
            self.u2 = DoubleConv(base * 4 + base * 2, base * 2)
            self.u1 = DoubleConv(base * 2 + base, base)
            self.out = nn.Conv2d(base, out_ch, 1)

        def forward(self, x):
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            c4 = self.d4(self.pool(c3))
            u = nn.functional.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            u = self.u3(torch.cat([u, c3], 1))
            u = nn.functional.interpolate(u, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            u = self.u2(torch.cat([u, c2], 1))
            u = nn.functional.interpolate(u, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            u = self.u1(torch.cat([u, c1], 1))
            return self.out(u)

    return UNet()


def dice_coef(logits, target, eps: float = 1e-6) -> Any:
    import torch

    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return num / den


def bce_dice_loss(logits, target) -> Any:
    import torch
    from torch.nn import functional as F

    return F.binary_cross_entropy_with_logits(logits, target) + (1.0 - dice_coef(logits, target))


def load_unet(weights: str | Path, device: str = "cpu"):
    import torch

    model = build_unet()
    ckpt = torch.load(str(weights), map_location=device, weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_mask(weights: str | Path, image_bgr: np.ndarray, size: int = 512, device: str = "cpu") -> np.ndarray:
    import torch

    h, w = image_bgr.shape[:2]
    rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    resized = _resize_keep(rgb, size)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)[None]).to(device)
    model = load_unet(weights, device)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob >= 0.5).astype(np.uint8) * 255
    return _resize_gray(mask, w, h)


def _resize_keep(img: np.ndarray, size: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    scale = size / float(max(h, w))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out[:nh, :nw] = small
    return out


def _resize_gray(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    side = mask.shape[0]
    scale = max(w, h) / float(side)
    nw, nh = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
    crop = mask[:nh, :nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
