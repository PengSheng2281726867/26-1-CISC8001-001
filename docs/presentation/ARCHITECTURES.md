# 实验模型架构图

每张图都从本项目的实现、配置和运行记录核对后绘制。同一模型在不同示范规模和检查点下共用一张图。

**ImageGen PNG** 使用原架构图作为参考，版面按宽屏 PPT 处理。生成后逐张核对了输入、维度、连线与输出。原图 PNG 和 SVG 保留更细的实现标注，SVG 可用于编辑。 [ImageGen 提示词与参考图](materials/architecture_imagegen/prompts.json)

| PPT 位置 | 实验 | 图 | 对照依据 |
| --- | --- | --- | --- |
| 第 3 页或备份页 | **Lift ALT**，原始 ALT 与 10／30／50／100 条实验 | [ImageGen PNG](materials/architecture_imagegen/lift_alt.png) · [原图 PNG](materials/architectures/lift_alt.png) · [SVG](materials/architectures/lift_alt.svg) | 两路同时刻 RGB 共用 ResNet-18，融合 9 维本体状态后得到 128 维检索向量。主对照每次回放一个 7 维动作。[编码器](../../experiments/alt_robomimic_lift/src/alt_lift/encoder.py) · [Lift 比较](../../experiments/lift_box_dp/RESULTS.md) |
| 第 3 页或备份页 | **Lift DP‑CNN**，10／30／50／100 条实验 | [ImageGen PNG](materials/architecture_imagegen/lift_dp.png) · [原图 PNG](materials/architectures/lift_dp.png) · [SVG](materials/architectures/lift_dp.svg) | 两路同时刻 RGB 加 9 维本体状态，条件 1D U‑Net 生成 `16 × 7`，10 步 DDIM，执行首个动作。[模型代码](../../experiments/lift_box_dp/src/model.py) · [训练配置](../../experiments/lift_box_dp/experiments/exp_001_lift30/config.json) |
| 第 4 页 | **视觉 Push‑T ALT**，30／100 条基线 | [ImageGen PNG](materials/architecture_imagegen/pusht_alt.png) · [原图 PNG](materials/architectures/pusht_alt.png) · [SVG](materials/architectures/pusht_alt.svg) | 当前 RGB 和二维推杆坐标，经两个分支和融合 MLP 得到 128 维检索向量，再取 8 步示范动作。[编码器](../../src/alt/encoder.py) · [基线配置](../../configs/alt_pusht_base.yaml) |
| 第 5 页备份 | **Push‑T B1**，图像分支加位姿监督 | [ImageGen PNG](materials/architecture_imagegen/pusht_alt_b1.png) · [原图 PNG](materials/architectures/pusht_alt_b1.png) · [SVG](materials/architectures/pusht_alt_b1.svg) | 位姿头 `128 → 64 → 4`，只用于训练。正式评估仍用融合向量检索。[B1 编码器](../../experiments/fixed_alt_encoder/b1alt/encoder.py) · [B1 结果](../../experiments/fixed_alt_encoder/docs/interpretation.md) |
| 第 4 页 | **本项目视觉 Push‑T DP**，30／100 条短训练、100／186 条长训练 | [ImageGen PNG](materials/architecture_imagegen/pusht_dp.png) · [原图 PNG](materials/architectures/pusht_dp.png) · [SVG](materials/architectures/pusht_dp.svg) | 最近两帧 RGB 和推杆坐标，U‑Net 通道为 128／256／512，16 步 DDIM，生成 `16 × 2` 并执行当前时刻起的 8 步。[短训练配置](../../experiments/pusht_dp/config.json) · [长训练配置](../../experiments/pusht_dp/long_config.json) |
| 第 4 页备份 | **官方 Push‑T 视觉 CNN 权重复评** | [ImageGen PNG](materials/architecture_imagegen/pusht_dp_official.png) · [原图 PNG](materials/architectures/pusht_dp_official.png) · [SVG](materials/architectures/pusht_dp_official.svg) | 执行接口与本地 DP 相同，保留上游 512／1024／2048 通道和 100 步 DDPM。[官方复评记录](../../experiments/pusht_dp/evidence_long/official/reference.json) · [官方配置](https://github.com/real-stanford/diffusion_policy/blob/main/image_pusht_diffusion_policy_cnn.yaml) |
| 第 6–7 页 | **Push‑T 规则检索**，全局／相对几何、同边筛选和纯对齐 | [ImageGen PNG](materials/architecture_imagegen/pusht_geometry.png) · [原图 PNG](materials/architectures/pusht_geometry.png) · [SVG](materials/architectures/pusht_geometry.svg) | 该图是策略流程。几何距离排序、候选阈值和坐标变换均为固定规则，读取模拟器提供的真值 T 位姿。[几何策略](../../experiments/geometry_oracle_alt/geometry_policy.py) · [接触规则](../../experiments/pusht_contact_matching/RESULTS.md) · [动作迁移](../../experiments/pusht_action_transfer/RESULTS.md) |
| 第 6、8 页 | **Push‑T 残差 MLP**，无界、有界和衰减变体 | [ImageGen PNG](materials/architecture_imagegen/pusht_residual.png) · [原图 PNG](materials/architectures/pusht_residual.png) · [SVG](materials/architectures/pusht_residual.svg) | 8 维相对几何与 16 维历史动作目标拼接，经过 `24 → 64 → 64 → 16` 网络，预测 8 步二维修正。把修正加到物体坐标系中的动作先验，再转回世界坐标。衰减版把块内修正权重从 1 线性降到 0。[残差网络](../../experiments/pusht_residual/src/study.py) · [衰减实现](../../experiments/pusht_residual/src/network_study.py) |

B2-B 只有离线候选覆盖分析。动作执行长度消融沿用各自的基线架构。Lift 原始 ALT 的不同执行块长均通过同一记忆库取动作。Push‑T 的固定候选动作迁移采用对应的检索方法和坐标对齐流程。[完整实验目录](../../experiments/README.md)

## 重绘

从仓库根目录运行。使用现有环境中的 Matplotlib。

```powershell
$python = '..\DP\.conda\pusht\python.exe'
& $python -B docs/presentation/build_architectures.py
```
