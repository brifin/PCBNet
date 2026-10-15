#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""赛题六最终提交入口（纯 CV：YOLO + OCR + 图结构匹配）。

约定完全按 2606sample_submission：
  - 由 /submission/run.sh 调用：python3 /submission/agent.py -input <输入目录> -output <输出目录>
  - 每个图片输出 <图片名去后缀>_Pred.json 到 <输出目录>/final_results/
  - 另输出 decision_log.json（纯 CV：token 全 0）

环境：评测镜像自带 torch/ultralytics/opencv；OCR（paddle/paddleocr）由随包 pylibs 提供
（构建时离线安装，运行时追加 sys.path）。本脚本启动只做一次依赖自检打印（不联网）；
缺依赖时设 PCBNET_SELF_HEAL=1 才会尝试联网补装。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# OCR 依赖装在 /submission/pylibs（构建时离线安装）。**追加**到 sys.path 末尾：
# 镜像自带的 torch/ultralytics/opencv 优先，只有镜像缺的包（paddle 等）才从这里取。
_PYLIBS = HERE / "pylibs"
if _PYLIBS.is_dir():
    sys.path.append(str(_PYLIBS))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# 评测机驱动可能比镜像 torch 期望的旧（torch cu130 vs 驱动 CUDA 12.9），
# torch 初始化 CUDA 会打一堆 UserWarning。这里先静音，再用显式设备选择回退 CPU。
import warnings as _warnings  # noqa: E402

_warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
_warnings.filterwarnings("ignore", message=".*CUDA driver.*")
_warnings.filterwarnings("ignore", message=".*NVIDIA driver.*")

# (import 名, pip 包名, 是否必需)
DEPS = [
    ("numpy", "numpy", True),
    ("cv2", "opencv-python", True),
    ("yaml", "pyyaml", True),
    ("tqdm", "tqdm", True),
    ("torch", "torch", True),
    ("ultralytics", "ultralytics", True),
    ("paddle", "paddlepaddle", False),      # OCR：缺了位号全丢，分数会崩，尽量装上
    ("paddleocr", "paddleocr", False),
]


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def check_env() -> None:
    """打印运行环境与依赖清单（只读，不联网）。

    评测镜像已自带推理依赖，这里不做任何安装；若日志里出现 MISSING，
    说明该依赖确实缺失（可通过 PCBNET_SELF_HEAL=1 让 agent 尝试联网补装）。
    """
    print(f"[env] python={sys.version.split()[0]} pylibs={'yes' if _PYLIBS.is_dir() else 'no'}")
    models_dir = Path.home() / ".paddleocr"
    print(f"[env] ocr_models={models_dir} exists={models_dir.is_dir()}")
    missing: list[str] = []
    for mod, pkg, req in DEPS:
        ok = _has(mod)
        print(f"[env] {mod:12s} {'OK' if ok else 'MISSING':7s} ({pkg}{'' if req else ', 可选'})")
        if not ok and req:
            missing.append(pkg)
    if not missing:
        return
    print(f"[env] 缺失必需依赖：{missing}")
    if os.environ.get("PCBNET_SELF_HEAL", "") in ("1", "true", "yes"):
        pip = [sys.executable, "-m", "pip", "install", "--break-system-packages", "--no-cache-dir"]
        try:
            subprocess.run(pip + ["--timeout", "300", "--retries", "10"] + missing, check=True, timeout=3600)
            print("[env] 联网补装完成")
        except Exception as exc:  # noqa: BLE001
            print(f"[env] 联网补装失败：{exc}")
    else:
        print("[env] 需要补装时请设置环境变量 PCBNET_SELF_HEAL=1 后重跑")


def collect_images(in_dir: Path) -> list[Path]:
    if not in_dir.is_dir():
        return []
    flat = sorted(p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if flat:
        return flat
    seen, out = set(), []
    for p in sorted(in_dir.rglob("*"), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", default=os.environ.get("INPUT_PATH", ""), help="输入目录")
    parser.add_argument("-output", default=os.environ.get("OUTPUT_PATH", ""), help="输出目录")
    parser.add_argument("--skip-env-check", action="store_true", help="跳过依赖自检打印")
    args = parser.parse_args()

    if not args.input or not args.output:
        print("Error: Missing required arguments")
        print("Usage: python3 agent.py -input <input_dir> -output <output_dir>")
        return 1

    in_dir, out_dir = Path(args.input), Path(args.output)
    t0 = time.time()
    print(f"[agent] input_dir={in_dir} exists={in_dir.is_dir()}")
    print(f"[agent] output_dir={out_dir}")
    print(f"[agent] cwd={os.getcwd()} PCBOCR_DEVICE={os.environ.get('PCBOCR_DEVICE', '(auto)')}")

    if not args.skip_env_check:
        check_env()

    from pcbnet.config import load_config
    from pcbnet.export_json import export_contest_json
    from pcbnet.schema import empty_contest_json
    from pcbnet.stage1_vision import extract_vision
    from pcbnet.stage2_structure import build_structure

    cfg = load_config(str(HERE / "configs" / "submission.yaml"))
    weights = (cfg.get("models") or {}).get("yolo_weights") or ""
    print(f"[agent] yolo_weights={weights} exists={Path(weights).exists() if weights else False}")

    images = collect_images(in_dir)
    print(f"[agent] found {len(images)} images")
    if not images:
        print("[agent] WARNING: 输入目录里没有找到图片", flush=True)
        return 1

    final_results = out_dir / "final_results"
    final_results.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for i, img in enumerate(images, 1):
        case_id = img.stem
        dest = final_results / f"{case_id}_Pred.json"
        try:
            vision = extract_vision(img, cfg, visualize=False)
            contest = build_structure(vision, cfg)
            export_contest_json(contest, dest)
            ok += 1
            print(f"[agent] [{i}/{len(images)}] {img.name} -> {dest.name} ok")
        except Exception:  # noqa: BLE001 单图失败输出空结构，不拖垮整体
            fail += 1
            try:
                dest.write_text(json.dumps(empty_contest_json(), ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            print(f"[agent] [{i}/{len(images)}] {img.name} FAILED")
            traceback.print_exc(limit=6)
        sys.stdout.flush()

    decision_log = {
        "method": "yolo",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processed_cases": len(images),
        "token_usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0},
    }
    # README 要求 final_results/decision_log.json；样本 agent.py 写 output 根 —— 两处都写
    for p in (final_results / "decision_log.json", out_dir / "decision_log.json"):
        try:
            p.write_text(json.dumps(decision_log, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    print(f"[agent] done: {ok} ok / {fail} failed, {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
