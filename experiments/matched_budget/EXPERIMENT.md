# 固定更新次数下的 ALT 与 DP 对照

## 研究问题

在同一任务内，共享训练示范、闭环评估起点和优化器更新次数时，ALT 与 DP 的表现如何随示范规模变化？同样的现象能否在三个独立训练种子中复现？

此前的 Lift 正式测试中，10 条组用验证指标分别为两种方法选权重，30 条组用各自的预设最终权重。视觉 Push‑T 的初始对照也采用各自的训练与选模方案。本实验对每个任务预先规定一个训练预算。每个模型都用最后一次更新后的权重，DP 使用该步的指数移动平均（EMA）权重。

| 任务 | 独立示范数 | 每模型更新次数 | 同一起点集合 | 每局步数上限 |
| --- | --- | ---: | --- | ---: |
| Lift‑PH | 10／30／100 | 800 | 24 个验证起点，90000–90023 | 250 |
| 视觉 Push‑T | 30／100 | 1,000 | 50 个正式测试起点，100000–100049 | 300 |

每个条件独立训练三次，训练种子为 42、43、44。ALT 的 ResNet‑18 从 ImageNet 预训练权重初始化，并在任务数据上微调。DP 按原配方随机初始化。批大小设为 32。Lift 两种策略均舍弃不满 32 的训练小批次。Push‑T 的 ALT 舍弃尾批，DP 沿用保留尾批的实现，并记录实际采样窗口数。两项任务分别使用原有的嵌套示范 ID，扩大示范规模时保留小规模组的全部轨迹。冻结设计见 [protocol.json](protocol.json)。[Lift 示范划分](../alt_robomimic_lift/configs/lift_ph_split42.json) · [Push‑T 示范划分](../splits/pusht_v7_split42.json)

## 方法与指标

- **Lift‑PH** 使用两台相机的图像和当前机械臂状态。ALT 对观测做对比学习，用当前观测查找最相似的示范帧并执行该帧的单步动作。DP 学习去噪得到 16 步动作块，每次也只执行一步。方块超过桌面基准 4 厘米算达标，连续 20 个控制步达标算保持一秒，另记末步是否达标。[验证起点 8 条](../lift_box_dp/experiments/exp_002_lift10_epochs/validation_initial_states.json) · [追加验证起点 16 条](../lift_box_dp/experiments/exp_002_lift10_epochs/validation_extra16_initial_states.json)
- **视觉 Push‑T** 的观测为 RGB 和推杆二维位置。ALT 使用当前一帧，DP 沿用两帧相同模态的观测历史。两种策略都提交 8 步绝对推杆位置，之后再观测。报告每局最高归一化覆盖分、末步覆盖分与成功率。环境在覆盖率严格超过 95% 时成功终止。

固定的是每个任务内的更新次数、名义批大小、最终权重规则和示范划分。更新次数从接触任务示范开始计数。每次更新的耗时、实际遍历轮数、训练目标、模型容量和学习率日程由各自实现决定。同一更新预算下，增加示范会减少数据被完整遍历的次数，也会扩大 ALT 的记忆库。示范规模效应以这次嵌套数据为条件。两个任务的分数、预算与评估协议各自定义，跨任务对照按任务分别解释。

## 复现

准备 [Lift WSL 仿真环境](../../environments/robomimic_lift/README.md)、Lift HDF5 和 [Push‑T 基础配置](../../configs/alt_pusht_base.yaml) 所列的 Zarr、上游代码与预训练权重。从项目根目录运行以下命令。两个入口均支持 `--smoke`，使用两次训练更新和缩小的仿真评估来检查基本接口。

```powershell
$studyPython = '..\DP\.conda\pusht\python.exe'
foreach ($method in 'alt','dp') {
    & $studyPython -B experiments/matched_budget/run_lift.py $method
    & $studyPython -B experiments/matched_budget/run_pusht.py $method
}
& $studyPython -B experiments/matched_budget/export.py
```

Lift 权重、记忆库和逐步动作留在 `experiments/lift_box_dp/artifacts/matched_lift_*`，Push‑T 权重、训练日志和评估记录留在 `artifacts/matched_budget/`。导出脚本核验配置、更新次数、示范配对和评估种子，Lift 还逐步核验达标记录。已核验的小型结果汇总见 [RESULTS.md](RESULTS.md)。
