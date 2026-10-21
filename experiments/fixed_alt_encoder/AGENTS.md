# AGENTS.md — Fixed ALT Encoder / B1 Pose-Supervised ALT-30

> **本文件只指导 `experiments/fixed_alt_encoder/` 内的实验。**
>
> 实验目的：在原 ALT-30 Push-T 基线之上，仅新增 **T 块位姿辅助监督（B1）**，验证是否能够让图像 Encoder 更准确地保留物体位置、旋转角度，从而提升最近邻检索和闭环控制表现。
>
> 请自主完成代码实现、测试、训练、离线诊断、正式评测、文档记录以及 **Git commit + push**。真实结果不佳也必须如实保存，严禁用未执行的实验填充报告。

## 0. 工作环境与作用范围

- **项目 / Git 仓库根目录**：`D:\Code\Course_code\COURSE-HUZI-CODE`
- **本次唯一的新增实验目录**：`D:\Code\Course_code\COURSE-HUZI-CODE\experiments\fixed_alt_encoder`
- **Git remote**：`https://github.com/PengSheng2281726867/26-1-CISC8001-001.git`
- **只读 ALT 参考**：`D:\Code\Course_code\ACl\alt`
- **只读 DP / Push-T 参考及原始数据**：`D:\Code\Course_code\DP\diffusion_policy`
- **已有的原始 ALT-Push-T 基线**：`src/alt/`、`src/pusht/`、`configs/alt_pusht_base.yaml`
- **已有 ALT-30 结果**：`experiments/runs/alt_030_split42_model42_pilot_v1/`
- **已有失败分析**：`docs/retrieval_failure_analysis.md`、`results/diagnostics/retrieval_failure_v1.json`

首先读取仓库根目录 `AGENTS.md`、`README.md` 和本文件。两级指令均需遵守。**本文件明确授权的唯一科研方法差异，是在本子实验训练阶段读取 T 块真实位姿作为辅助监督标签；真实位姿在闭环推理阶段绝不能作为策略输入。**

### 严格隔离

1. 除必要的已有模块 **import / 读取** 之外，所有本实验新增 Python、配置、测试、文档、小型结果图表必须放在 `experiments/fixed_alt_encoder/` 下。
2. 不要修改 `src/alt/`、`src/pusht/`、原始 `configs/`、已有实验记录、冻结 split manifest，或两份只读上游参考代码。若确实遇到无法适配的 API，先在实验目录实现 wrapper / adapter；除非得到用户进一步授权，不要改动共享基线。
3. 不创建嵌套 `.git`；使用项目根目录既有 Git 仓库。
4. 不覆盖任何旧 checkpoint、Memory Bank、日志、视频或同名实验目录；每次试验使用唯一 `run_id`。
5. 不重装原始环境，不擅自修改环境中的包；优先复用项目 README 中已验证的 Push-T Python 环境，记录环境版本。
6. 不能为获得更高 Score 调整测试 seed、评分函数、任务环境、数据集、动作执行方式。
7. 不训练 DP，不扩展到 ALT-100/300/500，不进行额外示范采集。

## 1. 研究问题与待验证假设

### 1.1 问题背景

现有 ALT-30 用 **一帧 96×96 RGB 图像 + 当前推杆 `(x_p, y_p)`** 生成 128D L2-normalized Embedding，然后在训练示范的 Memory Bank 中以余弦 Top-1 检索、直接执行后续 8 步绝对位置动作。

已有诊断提示：

- 修改输入的数值 `x_p, y_p` 对检索结果影响很弱；
- 不同的 T 块位置和角度可能产生很高的 Embedding 相似度；
- 错误匹配的绝对坐标动作可能导致停滞、重复检索；
- ALT-30 的官方 50-seed 平均 Push-T Score 约 **0.2744238994**。

这些是**先前实验观察，不是已经证明的 B1 机制因果结论**。

### 1.2 唯一主假设

> 在不改变输入模态、融合结构、Memory Bank、检索和动作执行机制的前提下，仅用训练集已有的 T 块真实位姿监督图像 Encoder，可以改善其物体几何表征，并可能进一步改善未见状态检索及 Push-T 闭环 Score。

务必区分：

- **Pose 预测更准**：图像分支更会识别 T 块；
- **检索几何误差更低**：最终融合 Embedding 更适合检索；
- **闭环 Score 更高**：找到的动作确实更有效。

三者**不能相互替代或必然推出**。如果 Pose 预测准确但检索/Score 未改善，这就是重要负结果。

## 2. 数据与评测协议：完全复用，不创建新划分

### 2.1 冻结数据范围

**本次不进行任何新数据划分。** 必须直接加载并校验现有：

`experiments/splits/pusht_v7_split42.json`

使用其中：

- `train_subsets["30"]`：唯一训练示范，**30 条完整 Episode**；
- `heldout_validation_episode_ids`：现有 **20 条**预留示范，仅用于辅助 Pose 指标与离线检索诊断；
- `test_env_seeds`：原有 **50 个**固定仿真评测 seed，正常为 `100000` 至 `100049`。

绝对不能：

- 用验证 Episode 训练 B1，或把验证帧放进 Memory Bank；
- 重新生成、洗牌、修改、重命名原 split；
- 改动既有训练示范、验证示范或评测初始状态；
- 以训练集上“库内向量找自己”的 99% 自检召回率，代替未见状态检索质量。

启动前执行并保存：split 文件 SHA256、数据集 fingerprint、精确训练/验证 Episode ID、评测 seed 列表。必须与已有 ALT-30 `dataset_manifest_ref.json` 一致；否则停止实验、说明差异。

### 2.2 真实位姿标签

从同一帧的原始 Zarr 数据读取：

```text
state[t] = [pusher_x, pusher_y, block_x, block_y, block_theta]
image[t] = 同一时间 t 的 RGB 图像
```

- `state[t, 0:2]` 是已有的模型输入 `agent_pos`。
- `state[t, 2:4]` 和 `state[t, 4]` **只作为 B1 的训练标签/离线诊断真值**。
- 保证 `(image_t, agent_pos_t, block_pose_t)` 是同一全局索引，且不跨 Episode 边界。
- 沿用现有 `PushTData`、`ObservationFrames` 的索引方式；在本实验目录新增只读 Dataset adapter，返回 `(aug1_image, aug2_image, agent_pos, block_pose_label)`，不要重写整个数据加载栈。
- 角度是弧度；位置单位是环境世界坐标（512×512），不要混用 96×96 像素坐标。

### 2.3 保持原有训练和评测设置

默认只做 **ALT-30 / model seed 42** 的一次独立 B1 Pilot：

| 项目 | 值 / 约束 |
|---|---|
| 输入 | 当前单帧 `image` + 当前 `agent_pos(x,y)` |
| ResNet | 同一 ImageNet 预训练 ResNet-18，全部参数可训练 |
| 图像特征 | 128D |
| 数值 xy 分支 | 原 `2 -> 64 -> 32` MLP |
| Fusion | 原 `160 -> 256 -> 128`，最终 L2 normalize |
| 对比 Loss | 原 NT-Xent，temperature = `0.4` |
| 数据增强 | 原轻量 ColorJitter，不新增/删除增强 |
| Batch size | `32`，沿用原 `drop_last` 等训练语义 |
| Optimizer | Adam，LR `1e-4` |
| Epochs | `40` |
| Model seed | `42` |
| Shuffle seed | `43` |
| Augmentation seed | `44` |
| Memory Bank | **仅**训练 30 条轨迹的所有有效帧 |
| 检索 | 128D Embedding 余弦 Top-1 |
| 动作 | 原绝对坐标 8 步 Chunk；无偏移补偿 |
| OOD | 无 threshold，无回退 |
| 官方评测 | 同一 50 seeds，max_steps `300`，`legacy_test=True` |

严格记录实际配置，不依赖此表代替运行证据。

## 3. B1 网络设计：只新增 Pose Auxiliary Head

### 3.1 原始主干不变

原 ALT-30：

```text
RGB image -> ResNet18 -> image_feature [B,128] --+
                                                      +-> concat -> Fusion MLP -> L2-normalized z [B,128]
agent_pos xy -> Position MLP -> pos_feature [B,32] -+
```

**B1 必须保持主干结构和 `z` 的定义不变**。新增旁路：

```text
                           +-> Pose Head -> [block_x, block_y, sin(theta), cos(theta)]
ResNet18 -> image_feature -+
                           +-> (与 Position MLP 特征融合，按原 ALT 构造 z)
```

建议 Pose Head 为小型 `Linear(128,64) -> ReLU -> Linear(64,4)`，使用配置明确记录维度。

**关键规则：**

1. Pose Head **只能接收图像分支的 `image_feature`**；不能接收 `agent_pos`、Fusion Embedding、真实 T 块状态、未来帧或动作。
2. 主干输入仍为 `image` 和 `agent_pos`，Pose Head 是**训练时的附加输出**，不是新增的推理输入。
3. 反向传播 Pose Loss 必须经过 Pose Head、ResNet 的输出映射层和 ResNet Backbone；不能 detach 图像特征。
4. 训练时一次前向应复用同一个图像特征来产生 `z` 和 `pose_pred`，避免额外的图像编码造成不必要的计算/随机性差异。
5. 对外推理接口必须继续兼容 `encoder(image, agent_pos) -> normalized_128D_z`，使原 `MemoryBank`、`ALTPolicy` 和官方 Runner 可直接复用。
6. 推理时 Pose Head 可以存在于 checkpoint 中但**不执行，也不参与相似度计算**。最终仍使用完整 `z[128]` 做余弦 Top-1，不能偷偷改为 geometry-weighted cosine。
7. 不在本次修改 Image/Position 分支的显式加权、归一化或融合层维度。这属于后续独立实验 A，不能和 B1 混合。

实现建议：在 `experiments/fixed_alt_encoder/` 编写独立 `B1FusionEncoder` / adapter，尽可能复用根目录 `src/alt/encoder.py` 的初始化及模块；不要直接改共享 `FusionEncoder`。

### 3.2 为了公平比较，控制初始化

- 每次 B1 训练都从 **同一 ImageNet 权重**重新初始化，不继续训练已有 ALT-30 checkpoint。
- 先创建/初始化与 ALT-30 **完全相同的 ResNet18、Image projection、Position MLP 和 Fusion MLP**，再新增 Pose Head；使用 model seed 42。
- 在正式训练前，比较已有 ALT-30 的初始化代码与 B1 初始化；确保新增 Pose Head 的随机初始化不改变前述原有层的初始权重。若无法直接核对当年的初始化 tensors，则使用重复实例化 + 相同 seed 的数值一致性测试说明可验证范围。
- 非 B1 部分的代码/设置差异必须归零；若不能归零，列出差异、评估混杂影响，不可声称这是“只加 Pose Loss”的严格消融。

## 4. Loss 定义和标签处理

### 4.1 Pose 标签

对同一图像帧对应的真实 T 块位姿：

```python
pose_target = [
    block_x / 256.0 - 1.0,
    block_y / 256.0 - 1.0,
    sin(block_theta),
    cos(block_theta),
]
```

原因：xy 映射到约 `[-1,1]`，角度的 sin/cos 处理 0 与 2π 的周期边界。

**禁止**把 `state[..., 2:5]` 当作推理输入，或在训练过程中把 `pose_target` 拼接到主干 Embedding。

### 4.2 两个独立增强视图共同接受相同 Pose 监督

每个训练样本：

```text
(image_aug1, image_aug2, same_agent_xy, same_block_pose)
```

前向得到：

```text
(z1, pose_pred1), (z2, pose_pred2)
```

定义：

```text
L_contrast = NTXent(z1, z2, tau=0.4)
L_xy      = 0.5 * (MSE(pred1[:2], target[:2]) + MSE(pred2[:2], target[:2]))
L_angle   = 0.5 * (MSE(pred1[2:], target[2:]) + MSE(pred2[2:], target[2:]))
L_pose    = L_xy + L_angle
L_total   = L_contrast + lambda_pose * L_pose
```

`MSE` 默认对所含元素取均值；两个视图的 Pose Loss **取平均，不因为视图数加倍其权重**。`L_xy`、`L_angle`、`L_pose`、`L_contrast` 和 `L_total` 必须分别记录。

主实验使用 **`lambda_pose = 1.0`** 作为事先固定的初始试验值；将其放在配置中，记录选择依据（归一化后的 Pose target 范围），不要依据测试 Score 临时调节。

训练后在验证集记录：

- T 块 xy 预测 L2 误差（还原到 512 世界坐标单位）；
- T 块旋转预测周期角度误差（度数，推荐 `atan2(pred_sin,pred_cos)` 解码）；
- xy/角度误差的均值、中位数、P95；
- NT-Xent 以及各 Pose Loss 曲线。

若梯度或 Loss 数值不合理，先在少量真实样本上调试代码/数值缩放，**不要自行加入其他监督任务或修改对比学习目标**。

## 5. 实验实施步骤与验收门槛

Codex 应按阶段自主完成，并在每个阶段保存进度；未通过门槛不得把结果标记为完成。

### Phase 0 — Read-only audit（不训练）

1. 检查当前 branch、remote、`git status` 和根目录 AGENTS 要求。
2. 读取已有 ALT-30 的 resolved config、数据/seed manifest、训练记录、Memory Bank 验证、官方评测结果及诊断。
3. 核对已冻结训练 30 个 Episode、验证 20 个 Episode、50 个仿真 seed，检查 split hash。
4. 核对 Push-T `state` 索引，确认 T 块位置和角度标签的来源与范围。
5. 保存 `docs/phase0_audit.md`；任何不一致必须记录并先修复实验目录内的适配问题。

### Phase 1 — 实现与测试（短跑，不算正式实验）

1. 建立实验目录内的独立 Dataset Adapter、`B1FusionEncoder`、训练入口、配置、离线诊断和评测 wrapper。
2. 实现 B1 的两视图前向、NT-Xent、Pose Loss 和 pose-only inference disable。
3. 测试 `image -> pose_pred` 无法访问数值推杆 xy：**固定图像改变 agent_pos，Pose Head 预测必须相同**（eval 模式）。
4. 测试训练 Pose Loss 的梯度确实传到 ResNet Backbone，且 Pose Head 更新。
5. 测试推理只输出 128D 单位范数 Embedding，能被原 `build_bank` 与 `ALTPolicy` 直接使用。
6. 验证原图像预处理、训练帧与 Pose 标签一一对齐，特别检查多个 Episode 的首尾帧。
7. 保证未加载或输出训练 Pose 标签的闭环推理接口正常运行。
8. 做一个极小 minibatch 的 sanity check，核实 Loss 有限、梯度非零、标签正确；它**不是**正式 40-epoch 结果。
9. 用 `pytest` / `unittest` 或现有项目测试方式运行自动测试，保存测试报告。

**门槛：** 关键测试全部通过，确认与原 baseline 唯一预期方法差异是训练 Pose Head / Pose Loss。

### Phase 2 — 完整 B1-ALT-30 Pilot（正式训练）

1. 使用 **冻结训练集 30 个 Episode**，model seed 42，40 epochs，Adam 1e-4，batch 32，NT-Xent tau 0.4。
2. 保存每 epoch 的训练 `contrast`, `xy`, `angle`, `pose`, `total` Loss，记录实际 optimizer steps、时间、显存、环境。
3. 最终 checkpoint 采用**预先固定的 final epoch**；不因为验证/测试表现挑选有利 epoch。
4. 从最终训练后 Encoder，使用原始未增强图像创建**全新 ALT-30 Memory Bank**；只允许训练 30 Episode 的帧进入 Bank。
5. 保存 checkpoint、Bank、配置快照和 SHA256；大型二进制放在 Git ignored `artifacts/`，不能直接加入 Git。
6. 保存独立 `run_id`，不要覆盖 `alt_030_split42_model42_pilot_v1`。

### Phase 3 — 未见帧检索与 Pose 诊断（正式闭环前）

使用**已有的同一批 20 个验证 Episode**查询 ALT-30 原基线 Bank 和 B1 Bank，必须尽量复用现有 `scripts/diagnose_representation.py` 的指标定义和相同查询样本/范围。

至少对比：

- **Pose Head 准确率**：预测 T 块位置误差、角度误差（在验证集）；
- **检索状态误差**：检索到的推杆 xy L2、T 块 xy L2、T 块角度周期误差（均值、中位数、P95）；
- **专家动作差异**：8 步绝对动作 Chunk RMSE（仅作为动作差异，不等同正确率）；
- **位置敏感性**：保持图像不变只替换数值推杆 xy，比较 top-1 改变比例和 `1-cos`，明确这是人工不一致输入的函数探针；
- **检索可视化**：同一验证帧，左右对比 ALT-30 原匹配和 B1 匹配，标注真实/匹配推杆、T 块位置/角度误差与相似度。

原始 ALT-30 训练/Bank/评测数据不得被重写。几何状态只用于**离线真值比较**，不用于 B1 策略实际检索。库内向量 Self-retrieval 自检可以做，但不可当作主要指标。

如果 Pose Head 预测很好，但几何检索误差并不改善，**如实记录“视觉几何识别与融合相似度脱节”**，仍可进入原定闭环评测；不要临时修改分支权重、相似度或输入特征。

### Phase 4 — 官方闭环 Push-T 评测

1. 重用根项目官方 `PushTImageRunner` 集成，不另造环境、分数或终止条件。
2. 输入仍只有一帧 RGB + 当前推杆 `(x,y)`；完全不把 `state[2:5]`、Pose Head 输出、奖励、未来信息传入检索策略。
3. 从 B1 的 **128D 融合 z** 检索 Top-1 训练 Bank，执行原始 8 步绝对动作，无补偿/无回退。
4. 运行已有相同 50 个固定 test seeds，`max_steps=300`，`legacy_test=True`。
5. 记录官方每回合 maximum reward、最终 reward、mean score、score std、置信区间（沿用原 bootstrap 定义）、完整动作检索轨迹、停滞/重复匹配统计。
6. 将原 ALT-30 的同一 50 seeds 指标作**paired comparison**：报告每 seed 的差值、平均差及 paired bootstrap 95% CI。
7. 保留少量预先指定的视频（默认既有前四个 seeds），不要依据成功/失败挑选有利案例。
8. 确认 eval Bank 使用的 Encoder checkpoint hash 等于本次训练的 checkpoint hash；拒绝混用不同模型或其他数据规模的 Bank。

若 Score 下降或变化不显著，必须在文档中清楚表述；**不能为了使 Score 好看而进行无记录的额外调参/重复测试**。

### Phase 5 — 分析与报告

写出清楚的因果链检查：

1. Pose 预测是否改善？
2. 检索到的真实 T 块位置与角度是否更接近？
3. 推杆 xy 分支敏感度是否仍然很弱？（B1 并未直接修复数值融合。）
4. 错误绝对动作 / 重复动作块现象是否缓解？
5. 闭环 Score 是否改善？

**允许且期待负结果。** 不要由单个 model seed 或置信区间覆盖 0 的实验断言方法必然有效/无效。明确区分证据、推测和下一步假设。

## 6. 建议的目录组织

全部新增的小型科研文件保持在本子目录，适当复用根目录模块：

```text
experiments/fixed_alt_encoder/
├── AGENTS.md                        # 本文件
├── README.md                        # 目的、方法、运行命令、结果导航
├── configs/
│   └── b1_pose_alt30_seed42.yaml     # 唯一正式 pilot 配置，包含 lambda_pose
├── src/
│   ├── __init__.py
│   ├── data_adapter.py              # 原冻结 30 集 + 训练姿态标签
│   ├── encoder.py                   # 原 FusionEncoder + image-only Pose Head
│   ├── loss.py                      # Pose Loss 与 NT-Xent 的组合
│   ├── train.py                     # 独立训练与完整日志
│   ├── diagnose.py                  # 验证集离线 Pose / 检索指标
│   └── evaluate.py                  # 复用官方 Runner 的 B1 adapter
├── scripts/
│   ├── check.py                     # 数据 / 兼容性 / 自动测试
│   ├── run_pilot.py                 # 训练 -> Bank -> 验证 -> 正式评测
│   └── compare_baseline.py          # 与原 ALT-30 逐 seed 对照
├── tests/
│   ├── test_pose_labels.py
│   ├── test_b1_model.py
│   ├── test_gradient_and_loss.py
│   └── test_baseline_compatibility.py
├── docs/
│   ├── phase0_audit.md
│   ├── experiment_log.md
│   ├── failure_log.md
│   ├── method_and_control.md
│   └── final_analysis.md
├── runs/
│   └── <unique_run_id>/             # 仅小 JSON / CSV / Markdown 报告
└── results/
    ├── comparison.csv
    ├── comparison.json
    └── figures/                    # 小型离线误差图和匹配示意
```

此结构可按实际需要微调，但**只允许变更本子目录**；不能以“改动小”为由在根目录 `src/` 中直接修改基础代码。

大型模型、完整检索日志、视频，统一放至仓库根目录已被 `.gitignore` 忽略的：

`artifacts/fixed_alt_encoder/<unique_run_id>/`

轻量 `runs/<run_id>/` 中保存重现用的信息：已解析配置、数据/代码/模型 SHA256、训练/验证/测试指标、artifact 路径、运行状态和小样本日志。注意已有 `.gitignore` 不会自动忽略本子目录的任意大文件；提交前必须检查大小和敏感内容。

## 7. 可复现性：每次运行必须记录

至少保存：

- 唯一 `run_id`、UTC 时间、当前 Git SHA、clean/dirty 状态与源文件 hash；
- 父实验根项目实际使用的 config、源代码及诊断依据；
- 数据 fingerprint、冻结 split SHA256、train-30 Episode ID、20 验证 ID、50 评测 seeds；
- ImageNet 预训练文件 hash，B1 checkpoint hash，B1 Memory Bank hash；
- ResNet/Fusion/Position MLP/Pose Head 结构；`lambda_pose` 与各 Loss 定义；
- 初始化 / shuffle / augmentation / 评测 seed；优化器、batch、epochs、step 数；
- Loss、Pose 验证误差、检索误差、官方闭环 Score 和逐 seed 指标；
- 执行命令、CUDA/GPU、Python/Torch 版本和主要依赖版本；
- 失败命令、异常、根因判断、修复方案、验证证据及后续状态。

配置文件和结果必须可在不覆盖旧运行记录的前提下复跑。绝不能仅写“与 baseline 相同”而省略关键配置来源和校验信息。

## 8. 自主排错与迭代规则

遇到失败不要立即停止，也不要默默改成另一个实验：

1. 阅读 traceback、定位最小可复现问题，确认是依赖、路径、数据、张量形状、训练数值、API 或 Git 问题。
2. 首先使用读取、单元测试、短批次验证；尽量不启动无意义的完整 40 epoch 重训。
3. 仅在 `experiments/fixed_alt_encoder/` 中修复本实验实现/兼容问题。需要改动科学方法（Loss 定义、`lambda_pose`、输入、采样、网络、动作机制、评测）时**不得把其结果继续命名为同一 B1 baseline**；应记录提议并等待用户授权新实验。
4. 每次重要失败追加至 `docs/failure_log.md`：UTC 时间、命令、异常、原因、改动、测试结果、是否影响既有可比性。
5. 记录失败/中断的实际运行状态，不编造训练成功、结果提升或 Git Push 成功。
6. 若大型训练受 GPU/环境限制，可先保存已通过的阶段与待办，不要删掉失败痕迹。

## 9. Git 维护（必须执行，不只保存本地）

本实验需要**实际 commit 并 push 到既有 GitHub 远端**。按阶段保存，不要把全部工作挤成一个含糊提交。

### 第一次检查

从仓库根目录执行：

```powershell
git status --short
git branch --show-current
git remote -v
git log -5 --oneline
```

如果有用户未提交的改动，不擅自删除、reset、stash 或覆盖；只 stage 本实验明确归属的文件。如果当前分支不能安全更新远端，说明情况，选择独立实验分支并 push 该分支，避免污染其他工作。

### 推荐提交里程碑

1. `exp(b1): add isolated pose-supervision experiment scaffold`
2. `exp(b1): implement pose head and supervised training with tests`
3. `exp(b1): record ALT-30 pose-supervised pilot and offline retrieval diagnostics`
4. `exp(b1): report paired official Push-T evaluation and analysis`

实际提交前：

- `git status --short`，`git diff --check`，执行有关自动测试；
- 严格审查 staged 文件：只包含本实验允许的文件，且不包含 checkpoint、视频、Zarr、`.env`、完整大日志或私密路径凭据；
- 优先使用 **显式路径** stage：`git add experiments/fixed_alt_encoder/`，不要用 `git add -A` 把无关项目改动带入；
- commit 后进行普通 `git push`（如首次在实验分支上则 `git push -u origin <branch>`）；
- 检查 push 输出、Git remote 对应 SHA；**未经真实确认不能说已上传**；
- 若认证/网络失败，保留本地提交，记录失败并重试合理次数；不能改写远端历史，也不能未经允许修改本机 Git 全局安全/凭据配置。

根目录 `AGENTS.md` 与现有实验报告均保持不变；只在本子目录中添加对它们的引用链接。

## 10. 最终交付与验收标准

完成后应有一个 GitHub 上可审阅的**独立 B1-ALT-30 实验目录**，包含以下可验证结果：

- `README.md`：可执行运行命令、环境说明、研究问题和代码入口；
- B1 代码、清晰的 Loss 定义、配置及通过的必要测试；
- 完整 40 epoch 训练记录、最终 checkpoint / Bank 的本地保存位置与 SHA；
- 在相同预留示范上的 Pose 预测、状态检索、动作差异统计和可视化；
- 官方同 50 seeds 闭环评测和对原 ALT-30 的 paired comparison；
- 科研解释：为何 B1 有效或无效、局限、可能原因及下一实验建议；
- 分阶段 Git commits 和每一次成功 push 的 commit SHA；
- 没有重划分数据、没有修改源 ALT/DP、没有污染主基线或偷用真实 T 块位姿进行推理。

最终给用户一个**简洁中文总结**：实验是否完成、关键 Pose / retrieval / Score 数字、相对 ALT-30 改善还是下降、Git commit/branch/push 状态，以及尚未解决的问题。

**开始时不要先长篇讨论方案，直接执行 Phase 0 审计 → Phase 1 实现与验证 → Phase 2 ALT-30 新模型训练 → Phase 3 未见帧诊断 → Phase 4 官方闭环评测 → Phase 5 报告与 Git 推送。不要开展新的数据划分或其他数据规模实验。**
