#!/bin/bash
# 本地自测（不依赖 docker）：解 zip → 按样本约定跑 agent.py -input/-output → 检查输出。
# 用法：bash submission/test_local.sh [zip路径] [测试图片目录] [python解释器]
# 建议 PY=.venv/bin/python（含 torch/ultralytics）；本机 paddle 与 torch 冲突时
# 会拷贝仓库 outputs/ocr_cache 作预生成 OCR 缓存（按图片绝对路径哈希命中）。
set -e
cd "$(dirname "$0")/.."
ZIP="${1:-dist/pcbnet_submission.zip}"
IMAGES="${2:-data/Dataset200/10gt}"
PY="${PY:-.venv/bin/python}"

WORK=$(mktemp -d /tmp/pcbnet_test.XXXXXX)
echo "[test] 解包 $ZIP -> $WORK"
unzip -q -o "$ZIP" -d "$WORK/app"
OUT="$WORK/out"
mkdir -p "$OUT"

if [ -d outputs/ocr_cache ]; then
  echo "[test] 拷贝预生成 OCR 缓存（本地自测用，不改变 zip 内容）"
  mkdir -p "$WORK/app/outputs"
  cp -r outputs/ocr_cache "$WORK/app/outputs/ocr_cache"
fi

echo "[test] 运行入口（样本约定）：$PY $WORK/app/agent.py -input $IMAGES -output $OUT"
"$PY" "$WORK/app/agent.py" -input "$IMAGES" -output "$OUT"

echo "[test] final_results 里 JSON 数量：$(ls "$OUT/final_results"/*.json 2>/dev/null | wc -l)"
echo "[test] decision_log："
cat "$OUT/final_results/decision_log.json" 2>/dev/null || cat "$OUT/decision_log.json" 2>/dev/null
echo "[test] 抽查第一个 _Pred.json 的前 40 行："
first=$(ls "$OUT/final_results"/*_Pred.json 2>/dev/null | head -1)
[ -n "$first" ] && head -40 "$first" || echo "（没有输出，见上方日志）"
echo "[test] 测试产物目录：$WORK"
