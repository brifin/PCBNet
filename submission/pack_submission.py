#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键生成赛题六最终提交 zip（在仓库根目录运行：bash submission/pack.sh）。

**自动打包**：不依赖固定的模块清单——
  1. 从 agent.py 实际 import 的入口（stage1_vision / stage2_structure / export_json / schema / config）
     出发，静态扫描 `from .xxx import` 自动收集**完整推理闭包**（含函数内懒加载的模块）；
  2. 被列入 EXCLUDED 的模块（可视化实现、LLM、评测器等）自动换成空实现/精简版，并打印提示；
  3. 扫描闭包源码里的知识库文件名引用，自动把对应数据文件带上（10gt 路径自动改写为 data/kb）；
  4. 权重路径自动从 configs/default.yaml 的 models.yolo_weights 解析（--weights 可覆盖）；
  5. OCR 依赖 wheels + OCR 模型随包，Dockerfile 构建时离线安装到 /submission/pylibs。

因此**无论 pcbnet/ 里的推理代码以后怎么改**，只要入口仍是 agent.py 的五个 import，
重新跑一遍 pack.sh 就会把最新的核心代码打进 zip。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "submission"

# --------------------------------------------------------------------------- #
# 入口与排除
# --------------------------------------------------------------------------- #
# agent.py 直接 import 的推理入口（自动收集闭包的起点）
ENTRY_MODULES = ["stage1_vision", "stage2_structure", "export_json", "schema", "config"]

STUB_VISUAL = '''"""可视化空实现（提交版：推理不画图，避免 matplotlib 依赖）。"""

from __future__ import annotations

import numpy as np


def run_vision_visualize(*a, **k):
    pass


def run_gt_visualize(*a, **k):
    pass


def run_wire_visualize(*a, **k):
    pass


def visualize_mask(*a, **k):
    pass


def visualize_skeleton(*a, **k):
    pass


def visualize_nodes_obstacle(*a, **k):
    pass


def visualize_grid(*a, **k):
    pass


def flip_gt_y(gt: dict, h_img: int) -> dict:
    return gt


_ = np
'''

STUB_GT_ADAPTER = '''"""类别映射（提交版只保留推理需要的 load_category_map）。"""

from __future__ import annotations

from pathlib import Path


def load_category_map(path=None):
    import yaml

    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "category_map.yaml"
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in data.items()}


__all__ = ["load_category_map"]
'''


def _generic_stub(name: str) -> str:
    return f'''"""提交版占位模块：`{name}` 在评测推理中用不到（被排除），仅保证 import 不报错。"""

__all__ = []
'''


# 被排除模块 -> 替换内容（visual/gt_adapter 有真替换，其余是空占位并告警）
EXCLUDED = {
    "visual": ("stub", STUB_VISUAL),
    "gt_adapter": ("stub", STUB_GT_ADAPTER),
    "correct": None,          # 包目录：直接不打包
    "stage3_correct": None,
    "scorer": None,
    "dataset": None,
    "label_export": None,
    "oracle": None,
    "unet": None,
}

# 源码里的知识库文件名 -> (源路径, 打包路径)；扫描闭包文本，引用了哪个就带哪个
DATA_REF = {
    "category_map.yaml": ("data/category_map.yaml", "data/category_map.yaml"),
    "yolo_classes.yaml": ("data/yolo_classes.yaml", "data/yolo_classes.yaml"),
    "type_priors.json": ("data/type_priors.json", "data/type_priors.json"),
    "type_value_priors.json": ("data/type_value_priors.json", "data/type_value_priors.json"),
    "type_list.md": ("data/Dataset200/10gt/type_list.md", "data/kb/type_list.md"),
    "BoxICLibrary_200cases.json": ("data/Dataset200/10gt/BoxICLibrary_200cases.json",
                                   "data/kb/BoxICLibrary_200cases.json"),
}

# 打补丁：把知识库路径从 data/Dataset200/10gt 改到 data/kb；去掉用不到的 matplotlib import
PATCHES = [
    ('_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Dataset200" / "10gt"',
     '_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"'),
    ('DEFAULT_LIB = Path(__file__).resolve().parent.parent / "data" / "Dataset200" / "10gt" / "BoxICLibrary_200cases.json"',
     'DEFAULT_LIB = Path(__file__).resolve().parent.parent / "data" / "kb" / "BoxICLibrary_200cases.json"'),
    ("from matplotlib import pyplot as plt\n", ""),
]

# OCR 运行时依赖（干净 venv 实测可用的最小集合；--no-deps 逐个装，不动镜像里的包）
OCR_PKGS = [
    "paddlepaddle", "paddleocr", "opencv-contrib-python", "numpy", "pillow", "pyyaml", "tqdm",
    "shapely", "pyclipper", "lmdb", "rapidfuzz", "fire", "fonttools", "requests", "setuptools",
    "opt_einsum", "protobuf", "decorator", "astor", "httpx", "urllib3", "idna", "certifi",
    "charset_normalizer", "termcolor", "anyio", "h11", "httpcore", "pydantic", "annotated_types",
    "typing_extensions", "scikit-image", "scipy", "imageio", "lazy_loader", "networkx", "tifffile",
    "packaging", "albumentations", "albucore", "simsimd", "stringzilla", "pydantic_core",
    "typing_inspection", "python_docx", "lxml", "beautifulsoup4", "soupsieve", "cython", "wheel",
]

DOCKERFILE = '''# 赛题六提交镜像构建文件（按 2606sample_submission 的 Dockerfile 约定）
# 平台会 docker build 本文件 -> ENTRYPOINT ["/submission/run.sh"] -> run.sh -input /input -output /output
FROM edarace-agent-base:ubuntu24

WORKDIR /submission

# 复制全部代码 / 权重 / 知识库 / OCR 依赖
COPY . /submission/

# OCR 依赖离线装到独立目录（--no-index 不联网；--no-deps 不动镜像里已有的包）
# 运行时由 agent.py 把 /submission/pylibs 追加到 sys.path，镜像自带包优先，互不干扰
RUN pip3 install --break-system-packages --no-cache-dir --quiet \\
      --no-index --no-deps --find-links=/submission/wheels --target=/submission/pylibs \\
      {PKGS}

# 预置 OCR 模型（运行时容器无外网，不能现场下载）
COPY ocr_models/.paddleocr /root/.paddleocr

RUN chmod +x /submission/run.sh

# 设备策略（评测机驱动较旧，镜像 torch 是 cu130，GPU 通常不可用）：
#   PCB_YOLO_DEVICE 未设 → 自动探测：能用 CUDA 用 GPU，否则回退 CPU（不报错、不刷 stderr）
#   PCBOCR_DEVICE=cpu   → OCR 固定走 CPU（paddle 是 CPU 版；也避免与 torch 抢 CUDA）
ENV PCBOCR_DEVICE=cpu

ENTRYPOINT ["/submission/run.sh"]
'''


# --------------------------------------------------------------------------- #
# 自动收集推理闭包
# --------------------------------------------------------------------------- #
_INTERNAL_IMPORT_RE = re.compile(r"^\s*from\s+\.([A-Za-z_]\w*)", re.M)


def discover_modules() -> tuple[list[str], set[str]]:
    """从入口模块出发，按 `from .xxx` 引用（含函数内懒加载）BFS 收集闭包。

    返回 (需打包的模块名列表, 出现在闭包里且被排除的模块名集合)。
    """
    queue = list(ENTRY_MODULES)
    seen: set[str] = set()
    excluded_seen: set[str] = set()
    while queue:
        name = queue.pop(0)
        if name in seen or name in excluded_seen:
            continue
        src = ROOT / "pcbnet" / f"{name}.py"
        if not src.exists():
            if (ROOT / "pcbnet" / name).is_dir():
                excluded_seen.add(name)   # 包目录（如 correct）：不打包
                continue
            print(f"[pack] WARNING 闭包引用但仓库里没有：{name}")
            continue
        seen.add(name)
        text = src.read_text(encoding="utf-8")
        for m in _INTERNAL_IMPORT_RE.findall(text):
            if m == name:
                continue
            if m in EXCLUDED:
                excluded_seen.add(m)
                continue
            if (ROOT / "pcbnet" / f"{m}.py").exists() and m not in seen:
                queue.append(m)
    return sorted(seen), excluded_seen


def discover_data_files(build_pcbnet: Path) -> tuple[list[tuple[str, str]], set[str]]:
    """扫描已拷贝的模块源码，找出引用到的知识库文件。返回 [(src_rel, dst_rel)] 与未命中的文件名。"""
    text = ""
    for p in build_pcbnet.glob("*.py"):
        text += p.read_text(encoding="utf-8") + "\n"
    files: list[tuple[str, str]] = []
    missing: set[str] = set()
    for fname, (src_rel, dst_rel) in DATA_REF.items():
        if fname in text:
            if (ROOT / src_rel).exists():
                files.append((src_rel, dst_rel))
            else:
                missing.add(fname)
    return files, missing


def resolve_weights(weights_arg: str) -> Path:
    """权重：--weights 参数 > configs/default.yaml 的 models.yolo_weights（相对仓库根解析）。"""
    if weights_arg:
        w = Path(weights_arg)
        return w if w.is_absolute() else ROOT / w
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")) or {}
    rel = (cfg.get("models") or {}).get("yolo_weights") or ""
    w = Path(rel)
    return w if w.is_absolute() else ROOT / w


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def stage(weights: Path, no_wheels: bool) -> Path:
    build = SUBMIT / "build"
    if build.exists():
        shutil.rmtree(build)
    (build / "configs").mkdir(parents=True, exist_ok=True)
    (build / "models").mkdir(parents=True, exist_ok=True)
    (build / "data" / "kb").mkdir(parents=True, exist_ok=True)

    # 1) 自动收集推理闭包
    modules, excluded_seen = discover_modules()
    dst = build / "pcbnet"
    dst.mkdir(parents=True, exist_ok=True)
    for name in modules:
        shutil.copy2(ROOT / "pcbnet" / f"{name}.py", dst / f"{name}.py")
    # 2) 被排除模块 -> 替换实现
    for name in sorted(excluded_seen):
        spec = EXCLUDED.get(name)
        if spec is None:
            print(f"[pack] 排除 `{name}`（推理用不到，不打包）")
            continue
        kind, content = spec
        (dst / f"{name}.py").write_text(content, encoding="utf-8")
        print(f"[pack] `{name}` 换成{'空实现' if kind == 'stub' else '占位'}（避免无关依赖）")
    # 3) 打补丁（路径改写 / 去 matplotlib）
    for old, new in PATCHES:
        hit = False
        for p in dst.glob("*.py"):
            text = p.read_text(encoding="utf-8")
            if old in text:
                p.write_text(text.replace(old, new, 1), encoding="utf-8")
                hit = True
        if not hit:
            print(f"[pack] NOTE 补丁未命中（核心代码可能已改，无碍）：{old[:48]!r}")

    # 4) 知识库数据文件（按闭包实际引用自动带）
    data_files, missing = discover_data_files(dst)
    for src_rel, dst_rel in data_files:
        d = build / dst_rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src_rel, d)
    if missing:
        print(f"[pack] WARNING 引用了但仓库缺失的数据文件：{sorted(missing)}")

    # 5) 权重
    if not weights.exists():
        raise SystemExit(f"权重不存在：{weights}（可用 --weights 指定）")
    shutil.copy2(weights, build / "models" / "yolo_best.pt")

    # 6) 提交版配置
    cfg = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    cfg = cfg.replace('yolo_weights: "outputs/models/yolo/train/weights/best.pt"', 'yolo_weights: "models/yolo_best.pt"')
    cfg = cfg.replace('unet_weights: "outputs/models/unet/best.pt"', 'unet_weights: ""')
    cfg += "\n# 提交版：禁用一切 LLM 能力；OCR 强制整图识别（2.x 检测器对紧裁剪不敏感）\n"
    cfg += "use_llm: false\nocr:\n  mode: image\n"
    (build / "configs" / "submission.yaml").write_text(cfg, encoding="utf-8")

    # 7) OCR wheels + 模型
    if not no_wheels:
        wheels = SUBMIT / "wheels"
        n_whl = len(list(wheels.glob("*.whl"))) if wheels.is_dir() else 0
        if not n_whl:
            raise SystemExit("submission/wheels 为空：先跑 bash submission/fetch_wheels.sh 下载 OCR 依赖")
        shutil.copytree(wheels, build / "wheels", dirs_exist_ok=True)
        print(f"[pack] OCR wheels {n_whl} 个（{sum(f.stat().st_size for f in wheels.glob('*.whl'))/1e6:.0f} MB）")
        ocr_models = Path(os.environ.get("PADDLEOCR_MODELS", "/data/luoxuan/.paddleocr"))
        if not ocr_models.is_dir():
            raise SystemExit(f"未找到 OCR 模型目录：{ocr_models}（用环境变量 PADDLEOCR_MODELS 指定）")
        shutil.copytree(ocr_models, build / "ocr_models" / ".paddleocr", dirs_exist_ok=True)
        print(f"[pack] OCR 模型（{sum(f.stat().st_size for f in ocr_models.rglob('*') if f.is_file())/1e6:.0f} MB）")

    # 8) 入口与 Dockerfile
    shutil.copy2(SUBMIT / "agent.py", build / "agent.py")
    shutil.copy2(SUBMIT / "run.sh", build / "run.sh")
    (build / "Dockerfile").write_text(DOCKERFILE.replace("{PKGS}", " ".join(OCR_PKGS)), encoding="utf-8")
    (build / "run.sh").chmod(0o755)
    return build


def zip_root(build: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(build.rglob("*")):
            if p.is_file() and "__pycache__" not in str(p):
                zf.write(p, p.relative_to(build).as_posix())
    print(f"[pack] zip 写好：{out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="YOLO 权重路径；默认自动读 configs/default.yaml")
    ap.add_argument("--out", default=str(ROOT / "dist" / "pcbnet_submission.zip"))
    ap.add_argument("--no-wheels", action="store_true", help="不打包 OCR wheels/模型（仅当镜像确定自带 OCR 时）")
    args = ap.parse_args()

    weights = resolve_weights(args.weights)
    print(f"[pack] 权重：{weights}")
    build = stage(weights, args.no_wheels)
    zip_root(build, Path(args.out))

    print("\n[pack] zip 内容：")
    for p in sorted(build.rglob("*")):
        if p.is_file() and "__pycache__" not in str(p) and "/wheels/" not in p.as_posix():
            print("   ", p.relative_to(build))
    if (build / "wheels").is_dir():
        print(f"    wheels/  ({len(list((build / 'wheels').glob('*.whl')))} 个 .whl)")
    print(f"""
[pack] 下一步：
  bash submission/test_local.sh "{args.out}"    # 本地跑通检查
  bash submission/test_docker.sh "{args.out}"   # 用基础镜像 docker build + run（需 sudo）
  # 提交：上传 {Path(args.out).name}（zip 根目录含 Dockerfile / run.sh / agent.py）""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
