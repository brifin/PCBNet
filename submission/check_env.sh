#!/bin/bash
# 探测评测基础镜像里到底有哪些依赖（30 秒出结果，决定要不要预置 wheels）
# 用法：bash submission/check_env.sh [镜像名]
BASE="${1:-edarace-agent-base:ubuntu24}"
echo "=== 探测镜像：$BASE ==="
sudo docker run --rm --entrypoint /bin/bash "$BASE" -c '
  echo "--- python ---"; python3 -V
  echo "--- 关键包 ---"
  for m in numpy cv2 yaml tqdm torch torchvision ultralytics paddle paddleocr matplotlib scipy pandas; do
    python3 - <<PY
import importlib.util as u
print(f"{'"$m"':14s}", "OK" if u.find_spec("'"$m"'") is not None else "MISSING")
PY
  done
  echo "--- pip 源 ---"; pip3 config list 2>/dev/null | head -5
  echo "--- 能否联网访问 pypi ---"; timeout 12 curl -sS -o /dev/null -w "pypi.org: %{http_code} %{time_total}s\n" https://pypi.org/simple/ || echo "pypi.org 不可达"
  echo "--- 预装包列表（部分）---"; pip3 list 2>/dev/null | head -40
'
