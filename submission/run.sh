#!/bin/bash
# 赛题六最终提交入口脚本（按 2606sample_submission 的约定：
# 平台以 ENTRYPOINT ["/submission/run.sh"] 启动，参数 -input <输入目录> -output <输出目录>）
set -e

INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 环境变量兜底
[ -z "$INPUT_DIR" ] && INPUT_DIR="${INPUT_PATH:-/input}"
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="${OUTPUT_PATH:-/output}"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 -input <input_dir> -output <output_dir>"
    exit 1
fi

echo "[agent] Input directory: $INPUT_DIR"
echo "[agent] Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR/final_results"

python3 /submission/agent.py -input "$INPUT_DIR" -output "$OUTPUT_DIR"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[agent] Completed successfully"
else
    echo "[agent] Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
