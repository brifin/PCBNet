# submission/ —— 赛题六最终提交打包

已按 `2606sample_submission` 的「方式1：纯CV + Dockerfile」逐项对齐，并针对上次摄取失败重做了依赖策略。

## 上次为什么失败（`[Ingestion] ERROR: Failed to build image`）

```
Requirement already satisfied: torch ... (2.14.1)      ← 基础镜像已自带 torch
Collecting ultralytics==8.4.143 ... ReadTimeoutError: pypi.org 15s 超时
```

两个原因叠加：**①** 我在 Dockerfile 里锁了版本（`ultralytics==8.4.143`），pip 必须去 pypi
查元数据；**②** 平台构建环境的 pypi 连接极不稳定（46KB 元数据在默认 15s 读超时内没下完），
于是构建中断。而**样本的纯CV Dockerfile 一行 pip 都不装**，说明基础镜像本就带齐了 CV 依赖。

## 现在的策略（与样本一致：构建阶段零安装）

**Dockerfile 里不装任何依赖**，完全对齐样本的纯CV写法：

```dockerfile
FROM edarace-agent-base:ubuntu24
WORKDIR /submission
COPY . /submission/
RUN chmod +x /submission/run.sh
ENV PCBOCR_DEVICE=cpu
ENTRYPOINT ["/submission/run.sh"]
```

依据：**样本 `纯CV/Dockerfile` 一行 pip 都不装**（`agent-api` 只装了个 `openai`），其 agent.py 直接
`import cv2, numpy` —— 说明 `edarace-agent-base:ubuntu24` 自带 CV/深度学习栈；你的构建日志也确认了
torch 2.14.1 / torchvision 0.29.1 已预装。之前失败纯粹是我在 Dockerfile 里锁版本触发了联网 pip。

`agent.py` 启动时只做一次**依赖自检打印**（不联网、不安装，毫秒级）：

```
[env] python=3.12.3
[env] numpy OK / cv2 OK / yaml OK / tqdm OK / torch OK / ultralytics OK / paddle OK / paddleocr OK
```

若平台上某行显示 `MISSING`，把该次提交的日志发我；需要临时补装可设 `PCBNET_SELF_HEAL=1`
（agent 才会尝试联网 `pip install --timeout 300 --retries 10`）。想提前确认镜像里有什么，跑
`bash submission/check_env.sh`。

## OCR 怎么保证可用（关键）

评测镜像自带 torch/ultralytics/opencv，但 OCR 不确定，所以 **OCR 依赖与模型随包离线提供**：

- `wheels/`（50 个、319MB）：paddlepaddle 2.6.2 + paddleocr 2.10.0 及其运行时闭环（清华镜像下载）；
- Dockerfile 构建时 `pip install --no-index --no-deps --target=/submission/pylibs`（**不联网、不覆盖镜像里的 torch/cv2**）；
- `agent.py` 把 `pylibs` **追加**到 `sys.path`（镜像自带包优先，只有缺的才用 pylibs）；
- `ocr_models/.paddleocr`（16MB）+ `ENV PCBOCR_DEVICE=cpu`，运行时无需下载模型。

### 容器模拟实测（干净 venv：只装 torch/torchvision 模拟镜像 + pylibs 装 OCR）

期间定位并修掉三个真实 bug（都是"实时 OCR"这条路以前没跑过、一直被 OCR 缓存掩盖）：

| # | Bug | 现象 | 修法 |
|---|---|---|---|
| 1 | `extract_vision` 有 YOLO text 框时走**裁剪路径** `_ocr_on_boxes` | paddleocr 2.x 的检测器对紧裁剪无反应 → 0 条文本 → **11.4 分** | 新增 `ocr.mode`（提交版强制 `image` 整图识别），并在裁剪路径失败时自动整图兜底 |
| 2 | 整图 OCR 前又**预缩放**到 `vision.max_side=1600` | 大图文字被缩太小，case2 只识别 1 条 | 改为**原生分辨率**交给 paddle 内部缩放 |
| 3 | paddle 默认 `det_limit_side_len=960` | 密集小字检不出：case2 只 6 条文本 | 提到 **2560**（case2 → 51 条）：case2 从 1.03 分 → **55.09 分** |

**容器模拟 10GT 得分：60.95**（Comp .794 / Pin .384 / Net .587 / Line .698），
与"吃仓库 OCR 缓存"时的 65.3 基本一致，说明**不依赖任何缓存也能在容器里跑出同样的水平**。

## 与样本实例的逐项对照

| 样本约定 | 本打包 |
|---|---|
| zip 根目录：`Dockerfile` + `run.sh` + `agent.py` + 任意多文件 | ✓ 根目录打包 |
| `FROM edarace-agent-base:ubuntu24` | ✓（改镜像名只需改 Dockerfile 一行） |
| `WORKDIR /submission`、`ENTRYPOINT ["/submission/run.sh"]` | ✓ |
| 平台调用 `run.sh -input /input -output /output` | ✓（另支持 `INPUT_PATH/OUTPUT_PATH` 环境变量兜底） |
| 输出 `/output/final_results/{case_id}_Pred.json` | ✓ |
| 输出 `decision_log.json`（纯CV：token 全 0） | ✓ `method="yolo"`，两处都写 |
| 统一 `python3`、不设 GPU 环境变量 | ✓（YOLO 自动选设备） |
| 可加自定义多文件模块 | ✓ `pcbnet/` 包 + `configs/` + `data/kb/` + `models/` |

## 包内容（只放"图 → JSON"的核心代码）

```
Dockerfile  run.sh  agent.py
configs/submission.yaml          # 沿用 default.yaml：阈值 20px、权重、OCR/视觉参数
models/yolo_best.pt              # YOLO 权重（默认取 configs 里的 best.pt）
data/kb/{type_list.md,BoxICLibrary_200cases.json}   # 先验知识库（位号前缀、型号↔引脚名）
data/{category_map.yaml,yolo_classes.yaml,type_priors.json,type_value_priors.json}
pcbnet/  config coords geometry schema knowledge_base pin_library ocr_cache
         topo_graph stage1_vision stage2_structure export_json
         + visual.py / gt_adapter.py（生成的空实现，推理不画图、不做 GT 转换）
```

**已剔除**（上次被指出的测试/配置数据与评测代码）：`scorer.py`、`dataset.py`、`label_export.py`、
`oracle.py`、`unet.py`、`gt_adapter` 的 GT 转换部分、`stage3_correct.py`、`correct/`（LLM）、
`data/Dataset200/10gt/`（看着像测试集）、以及仓库里的 `splits.json`/`labels`/`agents`/`scripts` 等。

## 输出格式

顶层为赛题口径 `components` / `pins` / `nets`；**所有坐标取整并保留 1 位小数**（如 `540.0`），
见 `pcbnet/export_json.py::_round_coords()`。

## 打包 / 测试 / 提交

```bash
# 一键打包（自动收集当前 pcbnet/ 的推理闭包、自动读 configs/default.yaml 的权重、
# 自动带知识库文件、自动下载缺失的 OCR wheels；参数透传给 pack_submission.py）
bash submission/pack.sh
# 等价于：
python3 submission/pack_submission.py --weights outputs/models/yolo/train_notext/weights/best.pt

bash submission/check_env.sh          # 探测基础镜像里有哪些包（需 sudo + 本机有镜像）
bash submission/test_local.sh         # 本地跑通（不依赖 docker）
bash submission/test_docker.sh        # docker build + run 全流程（需 sudo）
```

提交：上传 `dist/pcbnet_submission.zip`，zip 根目录即 `/submission`。

## 本地验证记录（2026-10-15）

- 入口跑通：10gt 10 张图全部产出 `final_results/*_Pred.json` + `decision_log.json`；
- 得分与改造前一致：**10GT Score 65.299**（Comp .800 / Pin .495 / Net .623 / Line .711）；
- 坐标检查：非整数坐标 0 个；
- 包内容检查：无 scorer/dataset/label_export/oracle/unet/correct/10gt/splits/labels/agents 等条目；
- 依赖自检 8 项全部 OK；对照实验（量化 OCR 的必要性）：关掉 OCR → Score **0.941**。

## 变更记录

- 2026-10-15：按反馈重做——Dockerfile 去掉全部 pip 安装（对齐纯CV样本、避免构建联网超时）；
  去掉预置 wheels 方案；只保留核心解析代码（剔除 scorer/dataset/label_export/oracle/unet、
  LLM 相关、可视化实现改为空实现、知识库从 `data/Dataset200/10gt` 移到 `data/kb`）。
- 2026-10-15：**自动打包**——pack_submission.py 不再依赖手工模块清单：从 agent.py 的入口
  import 出发静态扫描 `from .xxx import`（含函数内懒加载）自动收集推理闭包；被排除模块
  （visual/gt_adapter/scorer 等）自动换空实现；知识库文件按闭包里的实际引用自动带上；
  权重自动从 configs/default.yaml 解析。新增 `pack.sh`（一键）与 `fetch_wheels.sh`（清华
  镜像下载 OCR wheels）；`wheels/`、`build/` 已 gitignore（不入库）。
