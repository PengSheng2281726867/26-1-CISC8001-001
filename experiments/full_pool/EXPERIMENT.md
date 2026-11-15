# 全量示范和长训练对照

这个实验测量 ALT 与 DP 用完现有训练示范、经过多阶段训练后，能在 Lift‑PH 和视觉 Push‑T 上达到怎样的闭环表现。固定预算实验回答训练资源受限时的表现。全量实验给每种策略各自的训练预算，并用相同的验证起点挑选检查点。

| 任务 | 训练示范 | ALT 候选检查点 | DP 候选检查点 | 验证 / 正式测试 |
| --- | ---: | --- | --- | --- |
| Lift‑PH | 180 | 第 2、5、10、20、40 轮 | 第 10、40、80、120、150 轮 | 24 局 × 250 步 / 50 局 × 1,000 步 |
| 视觉 Push‑T | 186 | 第 2、5、10、20、40 轮 | 5,000、15,000、25,000、40,000 次更新 | 16 局 × 300 步上限 / 50 局 × 300 步上限 |

两种方法在同一任务使用完全一致的完整训练池。Lift 的 200 条和 Push‑T 的 206 条数据分别预留了 20 条轨迹，用于离线验证。两个任务的示范和仿真评价起点分开划分。这个实验每个方法和任务使用一个训练种子 42。Lift 和 Push‑T 的冻结原始划分中保存了精确的 `ordered_train_pool_episode_ids`。[Lift 划分](../alt_robomimic_lift/configs/lift_ph_split42.json) · [Push‑T 划分](../splits/pusht_v7_split42.json)

## 训练与选模

- **ALT** 微调 ImageNet 初始化的 ResNet‑18，用对比学习编码视觉和当前本体观测，批大小为 32。每个检查点用完整训练池独立重建记忆库，用余弦最相近的历史帧取专家动作。Lift 每次只执行一步，Push‑T 每次执行 8 步。光度增强沿用原分布，各样本和各视图独立抽样，使用四个固定种子的载入进程。
- **DP** 延用随机初始化的条件扩散 CNN 和 EMA 权重，批大小为 64。Lift 从单帧双摄像头图像和当前机械臂状态生成动作序列，每次执行一步。视觉 Push‑T 使用最近两帧 RGB 和推杆位置，每次执行 8 步。Push‑T 采用 16 步 DDIM 和 40,000 次更新预算对应的余弦学习率日程。
- **Lift** 先按一秒保持成功局数排序，依次以曾抬升、末步达标和较早轮数打破平局。方块高于桌面基准 4 厘米算达标。连续 20 控制步达标算保持至少一秒，末步达标另外记录。
- **Push‑T** 按每局最高归一化覆盖分的均值选检查点，平局取较早的检查点。覆盖率严格超过 95% 为成功。最高分衡量曾达到的覆盖，末步分衡量结束状态。视觉组输入为 RGB 和推杆位置。

每个任务仅在正式测试集评估验证挑出的检查点。训练长度与最佳检查点位置都在结果中报告。验证仍在上限处提高时，报告所测预算下的最好成绩。这里比较的是现有数据和两套实际训练配方所达到的表现。

## 复现

在仓库根目录准备 [Lift 环境](../../environments/robomimic_lift/README.md) 和 [Push‑T 路径配置](../../configs/alt_pusht_base.yaml) 所列的依赖、数据和权重。

```powershell
$fullPoolPython = '..\DP\.conda\pusht\python.exe'
& $fullPoolPython -B experiments/lift_box_dp/src/train_alt10.py full_pool_lift_alt_v1 --full-pool
& $fullPoolPython -B experiments/lift_box_dp/src/train.py full_pool_lift_dp_v1 --full-pool
& $fullPoolPython -B experiments/full_pool/run_lift.py alt
& $fullPoolPython -B experiments/full_pool/run_lift.py dp
& $fullPoolPython -B experiments/full_pool/run_pusht_alt.py
& $fullPoolPython -B experiments/pusht_dp/run.py study --run-id full_pool_upper_v1 --config experiments/full_pool/pusht_dp.json
& $fullPoolPython -B experiments/full_pool/export.py
```

大文件保存在 `experiments/lift_box_dp/artifacts/full_pool_lift_*`、`artifacts/full_pool/` 和 `artifacts/pusht_dp/full_pool_upper_v1/`。报告包含已选权重、所有候选的闭环验证成绩和正式测试的逐局数据。[结果](RESULTS.md) · [日志](LOG.md)
