import os
import re
import random
import subprocess
import sys
from pathlib import Path

# ==================== 驱动脚本：逐张随机运行 extract_single.py ====================
# 只做两件事：
#   1. 从 outputs/components/amp_3pin ~ l（不含 box）随机选图，每类 min(10, 实际数) 张；
#   2. 对每张图，把 extract_single.py 里「单个提取配置」的 target_comp_name 改成
#      该图片名（去 .png 后缀），然后运行 extract_single.py。
# 不改 extract_single.py 的任何其它代码。

PCBNET = Path("/data/luoxuan/PCBNet")
SRC_ROOT = PCBNET / "outputs" / "components"
EXTRACT_PY = PCBNET / "data_pipeline" / "extract_single.py"

# amp_3pin ~ l（不含 box），按字母序
TYPE_DIRS = [
    "amp_3pin", "amp_5pin", "ant", "battary", "bjt", "bjt_npn", "bjt_pnp",
    "block", "buzzer", "c", "circle_header", "crystal", "crystal_2pin",
    "crystal_3pin", "crystal_4pin", "d", "dc", "esd", "fuse", "gnd", "jumper", "l",
]
MAX_PER_TYPE = 10
SEED = 42


def set_target_comp_name(name: str):
    """把 extract_single.py 里 target_comp_name = "..." 改成 name（仅这一处）。"""
    src = EXTRACT_PY.read_text(encoding="utf-8")
    new_src, n = re.subn(
        r'target_comp_name\s*=\s*"[^"]*"',
        f'target_comp_name = "{name}"',
        src,
        count=1,
    )
    if n != 1:
        raise RuntimeError("未找到 target_comp_name 赋值行")
    EXTRACT_PY.write_text(new_src, encoding="utf-8")


def main():
    rng = random.Random(SEED)
    # 1. 随机选取：每类 min(10, 实际数) 张
    picked = []  # (type, name)
    for t in TYPE_DIRS:
        d = SRC_ROOT / t
        if not d.is_dir():
            print(f"[跳过] {t} 不存在")
            continue
        names = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".png"))
        k = min(MAX_PER_TYPE, len(names))
        chosen = rng.sample(names, k)
        for name in chosen:
            picked.append((t, name))

    print(f"共 {len(picked)} 张待处理（每类 ≤{MAX_PER_TYPE}，随机选取，seed={SEED}）")
    print("清单（前 10）:", [f"{t}/{n}" for t, n in picked[:10]])

    # 2. 逐张：改 target_comp_name → 运行 extract_single.py
    ok = fail = 0
    for i, (t, name) in enumerate(picked, 1):
        try:
            set_target_comp_name(name)
            r = subprocess.run(
                [sys.executable, str(EXTRACT_PY)],
                cwd=str(PCBNET),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode == 0:
                ok += 1
            else:
                fail += 1
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
                print(f"[{i}/{len(picked)}] 失败 {t}/{name}: " + " | ".join(tail))
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(picked)}] 异常 {t}/{name}: {e}")

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    # 恢复 target_comp_name 为占位值（避免留最后一张的状态）
    set_target_comp_name("")
    print("已恢复 target_comp_name 为占位空值")


if __name__ == "__main__":
    main()
