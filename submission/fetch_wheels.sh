#!/bin/bash
# 下载 OCR 运行时依赖 wheels（paddlepaddle 2.6.2 + paddleocr 2.10.0 及其闭环）。
# 清华镜像下载；torch/torchvision 用约束文件排除（基础镜像已自带，不重复下载）。
# 幂等：已存在的 wheel 不会重复下载。
set -e
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

mkdir -p submission/wheels
printf 'torch==2.14.1\ntorchvision==0.29.1\n' > /tmp/pcbnet_wheel_constraints.txt

echo "[wheels] 从清华镜像下载 OCR 依赖（约 300MB，首次 1~2 分钟）…"
"$PY" -m pip download \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  paddlepaddle==2.6.2 paddleocr==2.10.0 \
  -c /tmp/pcbnet_wheel_constraints.txt \
  -d submission/wheels \
  --only-binary=:all: --no-cache-dir

# 去掉重复的 opencv 变体（paddleocr 只用一个；省 ~130MB）
rm -f submission/wheels/opencv_python-*.whl submission/wheels/opencv_python_headless-*.whl

echo "[wheels] 完成：$(ls submission/wheels/*.whl | wc -l) 个 wheel，$(du -sh submission/wheels | cut -f1)"
