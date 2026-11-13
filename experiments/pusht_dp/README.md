# Push-T 视觉 ALT 与扩散策略对照

本实验比较两种使用 RGB 和二维推杆坐标的闭环策略。ALT 从训练轨迹匹配一帧并回放动作，DP 根据最近两帧生成动作。两者用相同的冻结训练轨迹和 50 个测试初始状态，每次执行 8 步动作，每局最多 300 步。

## 数据与模型

使用 [Push-T v7 冻结划分](../splits/pusht_v7_split42.json)。206 条独立示范中，20 条保留给离线验证，30 和 100 条训练组为嵌套子集。原计划的 300 和 500 条超过训练池容量，均标为 `NOT_FEASIBLE_WITH_CURRENT_DATA`。

DP 从头训练本地参考源码中的 `DiffusionUnetHybridImagePolicy`。它用 ResNet 图像编码与推杆位置条件预测动作噪声，保留 16 步生成长度，取从当前帧对齐位置起的 8 个绝对目标。相比本地上游的图像 CNN 配方，本次缩小 U-Net 通道至 128/256/512，用 100 个训练噪声时间步和 16 步 DDIM 采样，在一张 RTX 5090 上运行。训练与评估都只给 DP RGB 和推杆坐标。限幅归一化只拟合所选训练轨迹。训练窗口按上游采样规则在同一条轨迹边界复制动作和观测。

两个规模各从相同随机种子开始，AdamW、批次 64、最多 5,000 次更新，用指数移动平均（EMA）权重评估。1,000、2,500、5,000 次更新各在独立的 16 个初始种子 92000–92015 上评估，按平均最高覆盖得分选检查点，平局取较早者。正式测试固定 100000–100049。DP 保留两帧观测和独立选模，既有 ALT 保留单帧观测和第 40 轮权重。两种训练配方、预训练、批大小均有差异。

## 复现

从仓库根目录执行。环境、数据与 DP 源码见 [基础配置](../../configs/alt_pusht_base.yaml) 和 [版本来源](../../docs/upstream_provenance.md)。使用新运行 ID。脚本在读取时核对数据全量指纹、划分哈希、序列边界，评估使用上游 `PushTImageRunner.run` 和同步向量环境。

```powershell
$pushTPython = '..\DP\.conda\pusht\python.exe'
& $pushTPython -B experiments/pusht_dp/run.py smoke --scale 30 --run-id smoke_reproduction
& $pushTPython -B experiments/pusht_dp/run.py study --run-id split42_model42_reproduction
```

`train` 与 `test` 可分别指定 `--scale 30` 或 `--scale 100`。完整检查点、运行日志和视频位于仓库根目录的 `artifacts/pusht_dp/`。 [参数](config.json) · [评估与统计](RESULTS.md) · [实验记录](LOG.md) · [真值几何与小网络](../pusht_residual/README.md)

## 延长训练与使用全部训练池

[长预算配置](long_config.json) 比较 100 条与 186 条。186 条取冻结清单的完整训练池，保留 20 条预留示范及原来的测试种子。从头训练相同的精简 CNN，验证检查点为 5,000、10,000、15,000 次更新。两组采用相同的更新预算与学习率日程。长预算组和初版短预算组的余弦学习率日程长度随更新上限变化。

另复评上游视觉 CNN 的一个公开检查点，保持其原有宽度、EMA 权重和 100 次 DDPM 去噪，统一使用本仓库的 50 个测试种子。该检查点的训练数据划分来自上游，用于核对评估与量级。训练轨迹完全匹配的 ALT 对照仍采用本仓库的 30 条与 100 条视觉策略。

```powershell
$pushTPython = '..\DP\.conda\pusht\python.exe'
& $pushTPython -B experiments/pusht_dp/run.py study --config experiments/pusht_dp/long_config.json --run-id long_split42_model42_reproduction
& $pushTPython -B experiments/pusht_dp/official_reference.py --output artifacts/pusht_dp/official_reference_reproduction
```

复评入口要求同级 DP 项目提供 SHA‑256 固定的[上游视觉 CNN 检查点](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch%3D0500-test_mean_score%3D0.884.ckpt)。论文图像策略比较见[表 II](https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf)。
