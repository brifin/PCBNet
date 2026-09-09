#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train YOLO on exported train/val labels. Does not use the test split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcbnet.config import load_config, project_root


def _resolve_model(name: str) -> str:
    local = Path(name)
    if local.exists():
        return str(local)
    for cand in (ROOT / name, ROOT / "outputs" / "models" / "yolo" / name):
        if cand.exists():
            return str(cand)
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--smoke", action="store_true", help="1 epoch / imgsz 640 sanity run")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics。建议在 Python 3.10–3.12 环境执行：")
        print("  py -m pip install ultralytics")
        print("官方包：https://pypi.org/project/ultralytics/")
        return 1
    
    cfg = load_config(args.config)
    train_cfg = ((cfg.get("train") or {}).get("yolo") or {})
    data = args.data_yaml or str(project_root() / "outputs" / "labels" / "dataset.yaml")
    if not Path(data).exists():
        print(f"缺少 {data}，先运行: py scripts\\prepare_labels.py --splits train,val")
        return 1
    
    epochs = 1 if args.smoke else int(args.epochs if args.epochs else train_cfg.get("epochs", 50))
    imgsz = 640 if args.smoke else int(args.imgsz if args.imgsz else train_cfg.get("imgsz", 1280))
    batch = 2 if args.smoke else int(args.batch if args.batch else train_cfg.get("batch", 4))
    project = project_root() / "outputs" / "models" / "yolo"
    project.mkdir(parents=True, exist_ok=True)
    model_name = _resolve_model(args.model or train_cfg.get("model") or "yolo11n.pt")

    if not Path(model_name).exists() and Path(model_name).name == "yolo11n.pt":
        bundled = project / "yolo11n.pt"
        if bundled.exists():
            model_name = str(bundled)

    kwargs = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": "smoke" if args.smoke else "train",
        "exist_ok": True,
        "patience": 20,
        "workers": 0,
        "plots": False,
    }

    if args.device:
        kwargs["device"] = args.device

    print(f"[yolo] model={model_name} data={data} epochs={epochs} imgsz={imgsz} batch={batch}")

    try:
        results = YOLO(model_name).train(**kwargs)
    except Exception as exc:
        print(f"训练失败: {exc}")
        print("若是下载权重被拦截，请把官方 yolo11n.pt 放到仓库根目录或 outputs/models/yolo/ 后再跑。")
        return 1
    
    save_dir = Path(getattr(results, "save_dir", project / kwargs["name"]))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    pointer = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best": str(best) if best.exists() else "",
        "last": str(last) if last.exists() else "",
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
    }
    
    (project / "last_run.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    print("训练完成。把下面路径填进 configs/default.yaml 的 models.yolo_weights：")
    print(f"  {pointer['best'] or pointer['last']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
