#!/bin/bash
# 一键生成赛题六最终提交 zip：自动打包当前 pcbnet/ 的推理代码 + 权重 + 知识库 + OCR 依赖。
# 用法：bash submission/pack.sh [额外参数透传给 pack_submission.py，如 --weights ...]
set -e
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

# 1) OCR wheels 缺失时自动下载（清华镜像，约 1~2 分钟）
if [ -z "$(ls submission/wheels/*.whl 2>/dev/null | head -1)" ]; then
  echo "[pack] wheels 缺失，自动下载 OCR 依赖…"
  bash submission/fetch_wheels.sh
fi

# 2) OCR 模型目录检查
OCR_MODELS="${PADDLEOCR_MODELS:-/data/luoxuan/.paddleocr}"
if [ ! -d "$OCR_MODELS/whl" ]; then
  echo "[pack] 警告：未找到 OCR 模型目录 $OCR_MODELS（OCR 将无法在容器内离线初始化）"
  echo "[pack]        用环境变量 PADDLEOCR_MODELS 指向含 whl/ 的目录，或 --no-wheels 跳过"
fi

# 3) 打包（权重自动读 configs/default.yaml；参数原样透传）
echo "[pack] 开始打包（权重自动从 configs/default.yaml 解析）…"
exec "$PY" submission/pack_submission.py "$@"
