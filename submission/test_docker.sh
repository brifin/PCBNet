#!/bin/bash
# 按 2606sample_submission 的方式用 docker 构建并自测（需要 docker 权限 + 本机有基础镜像）。
# 用法：bash submission/test_docker.sh [zip路径] [测试图片目录] [基础镜像]
set -e
cd "$(dirname "$0")/.."
ZIP="${1:-dist/pcbnet_submission.zip}"
IMAGES="${2:-data/Dataset200/10gt}"
BASE_IMAGE="${3:-edarace-agent-base:ubuntu24}"

WORK=$(mktemp -d /tmp/pcbnet_docker.XXXXXX)
unzip -q -o "$ZIP" -d "$WORK/app"
mkdir -p "$WORK/out"

echo "[test] 基础镜像：$BASE_IMAGE"
echo "[test] docker build ..."
sudo docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t pcbnet-submission:test "$WORK/app"

echo "[test] docker run（样本约定：-input /input -output /output）"
sudo docker run --rm \
  -v "$IMAGES:/input:ro" \
  -v "$WORK/out:/output:rw" \
  pcbnet-submission:test \
  -input /input -output /output

echo "[test] final_results JSON 数量：$(ls "$WORK/out/final_results"/*.json 2>/dev/null | wc -l)"
echo "[test] decision_log："
cat "$WORK/out/final_results/decision_log.json" 2>/dev/null || cat "$WORK/out/decision_log.json" 2>/dev/null
echo "[test] 测试产物目录：$WORK"
