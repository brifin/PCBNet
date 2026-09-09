# AI-driven automatic PCB schematic parsing

赛题六（PCB 原理图 → 结构化 JSON）的可打分基线，按 PCBnet 论文三阶段实现，并按赛题
权重（Component 0.3 / Pin 0.25 / NetHypergraph 0.35 / NetLine 0.1）打分。

数据目录默认：`f:/AI专项/112_竞赛题目/Dataset300`（当前 64 例）。本仓库不修改原始数据。

## 为什么之前是 0 分（已定位）

- 评分器没问题：GT 自比四项 F1 = 1.0（`eval.py --self-check`）。
- 真正原因是赛题的**级联硬匹配**：component `key` 必须完全相等，key 错则该元件的
  Component / Pin / 相关 Net 全部作废。
- 旧管线用整图 OCR + 轮廓回退，位号 key 基本对不上，导致 test 全 0。

## Oracle 阶梯（先定位泄漏，再谈提分）

`scripts/oracle_eval.py` 用 GT 逐层替换感知，隔离每一层的贡献：

| 层 | 注入 | 含义 |
|---|---|---|
| L0 | 真实管线 | 现状 |
| L1 | GT 元件框+类+文字，stage2 建 pin/net（几何 pin） | 结构构建上限 |
| L2 | L1 + GT pin tip/number | 加入完美引脚 |
| L3 | L2 + GT 导线骨架 | 加入完美导线 |
| L4 | pred = 全 GT | 必须 100，否则适配器/评分器与赛题不一致 |

当前结论（test 7 例）：

- **L4 = 100**：适配器与评分器和赛题格式一致。
- **L2/L3 原始分约 19，ConfidentScore ≈ 54**：在“可恢复位号”的子集上，完美感知 + 本
  仓库 stage2 能到 54 分，且 pin 通道相比几何 pin 提升约 +11。架构成立。
- 原始分被“无位号元件”拉低：这些 GT 用 `cpntId`（如 `C_15`）当 key，**无法从图像还原**。
  真实赛题原理图通常印有完整位号，天花板远高于本地 test。

`ConfidentScore`（见 `pcbnet/scorer.py: filter_confident`）只在“key 是真实位号”的子集上
计分，是本地开发的诚实指标；提交仍按完整 Contest Score。

## 架构

```text
Stage1 感知  -> Stage2 结构        -> Stage3 门控校正
components      Hungarian 绑字         低置信 token 才修
text (裁剪OCR)  pin 通道 + 吸附导线     规则(1/l/I,O/0) -> qwen3.7-plus(有预算)
pin_tip         骨架图建网 + 飞线并网
wires(U-Net)    两点 edges 导出
```

- 检测类别：粗元件类 + `text` + `pin_tip`（`data/yolo_classes.yaml`）。禁止把 text/pin
  当成 component（`stage1_vision._split_detections`）。
- OCR：优先按检测到的 text 框裁剪识别，竖排试转 90° 取高置信；无检测器时回退整图 OCR。
- 绑字：Hungarian + 三约束（前缀↔类别、横竖对齐、局部唯一）；value 保持原始大小写。
- 建网：pin tip 吸附导线端点，骨架连通域建网，同名非 VCC/GND 网络标签飞线并网。
- Stage3：默认关闭；只修低置信 token，`value` 不动；qwen 仅在配置了 endpoint 时调用。

## 安装

```bat
cd /d "E:\AICode\AI-driven automatic PCB schematic parsing"
py -m pip install -r requirements.txt
```

评测/推理若要 OCR，请用装了 PaddleOCR 的 **Python 3.12**：

```bat
py -3.12 scripts\eval.py --split test
```

## 常用命令

```bat
:: 1) 定位泄漏（不需要模型，随处可跑）
py scripts\oracle_eval.py --split test --levels L1,L2,L3,L4

:: 2) 导出三类检测标签 + 导线 mask（只用 train/val，绝不导 test）
py scripts\prepare_labels.py --splits train,val

:: 3) 训练（需 Python 3.10-3.12 + ultralytics / torch）
py -3.12 scripts\train_yolo.py           :: 元件+text+pin_tip 检测
py -3.12 scripts\train_unet.py           :: 导线分割

:: 4) 填回权重到 configs/default.yaml 的 models.yolo_weights / unet_weights 后
py -3.12 scripts\infer.py --split test
py -3.12 scripts\eval.py  --split test    :: Contest + Aligned + 四项 F1

:: 5) GT 自比（评分器正确性）
py -3.12 scripts\eval.py --split all --self-check
```

## 启用 Stage3（qwen3.7-plus）

默认关闭且离线安全（不配置 endpoint 就是 no-op，符合内网策略）。在赛题测试平台上：

```bat
set QWEN_BASE_URL=<平台提供的地址>
set QWEN_API_KEY=<平台提供的key>
```

并在 `configs/default.yaml` 设 `stage3.enabled: true`。`max_calls_per_case` / `time_budget_sec`
保证 30–50 例在 2 小时内完成（超时案例记 0）。

## 提分下一步（按性价比）

1. **训练检测器**：粗类元件 + text + pin_tip（YOLOv11s/m，imgsz 1280，大图 tile）。这是把
   L0 拉到 L2 附近的关键，纯 CPU 训不动，需 GPU。
2. **补数据**：64 例远不足以复现论文 94% mAP；赛题另有最多 +10 数据贡献分。
3. **训练 U-Net 导线**，替换形态学骨架，减少 NetLine 假阳。
4. Stage3 只清洗少量低置信 token，不要端到端喂大模型（论文 Table I：Qwen 连通性约 8%）。

## 输出

- `outputs/pred/<case>/<case>.json` 预测（仅 components/pins/nets，左下角坐标、两点 edges、
  `pin_<num>` 键、`hyperGraph` 括号格式）。
- `outputs/gt/<case>.json` 由 CVJsonStd 转换的赛题真值。
- `outputs/score_report.md`、`outputs/score.json`：Contest + Aligned + 四项 F1。
- `outputs/oracle_report.md`、`outputs/oracle.json`：阶梯诊断 + ConfidentScore。

单案：`Score = 100*(0.3*Comp + 0.25*Pin + 0.35*NetHyper + 0.1*NetLine)`；阈值 bbox 20px、
pin/line 5px。v0 不端到端调用大模型。
