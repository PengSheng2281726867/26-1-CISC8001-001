# 机器人模仿学习实验｜ALT 与 Diffusion Policy

本仓库研究机器人用少量专家示范学习闭环控制时，**直接检索专家动作**与**学习生成动作**各有什么优势。目前有两个模拟任务。Push-T 用图像和推杆位置控制 T 形物块，只研究动作查找表（ALT）。Lift-PH 用两路图像和机械臂本体状态抓取并抬升方块，在相同示范与初始状态下比较 ALT 和扩散策略（DP-CNN）。

## 结果入口

| 任务 | 已完成的比较 | 结果与复现 |
| --- | --- | --- |
| Lift-PH | ALT 与 DP 的 10、30 条正式测试，以及 10、50、100 条轮数验证 | [中文结果总览](experiments/lift_box_dp/RESULTS.md) · [实验索引](experiments/lift_box_dp/README.md) |
| Push-T | 视觉 ALT、位姿监督、几何规则、动作长度与坐标迁移 | [中文实验总览](docs/pusht_results.md) · [当前帧与训练候选图集](docs/pusht_matching_gallery.md) |

## 实验发现

- **Lift 的正式测试中，DP-30 更常完成短时抬升。** 保持至少 1 秒的成功数为 DP 49/50、ALT 34/50。10 条示范时为 DP 23/50、ALT 18/50，配对 95% 区间跨零。
- **Lift 的轮数验证显示训练阶段会改变排序。** 用 100 条示范，约 770 次更新时 ALT 和 DP 分别为 17/24、12/24。第 40 轮分别为 8/24、24/24。DP-100 第 150 轮同样是 24/24，但终点仅 6/24。保持 1 秒指连续 20 个控制步高于桌面基准高度加 4 厘米。验证每局 250 步，正式测试每局 1,000 步。[评估定义与逐轮结果](experiments/lift_box_dp/RESULTS.md)
- **Push-T 上未观察到明确的规模收益。** ALT 从 30 条增至 100 条，平均官方覆盖得分为 0.2744 和 0.2745，配对差的 95% 区间为 −0.0970 到 +0.0932。诊断发现相似图像有时检索到几何位置不匹配的动作。[检索失败分析](docs/retrieval_failure_analysis.md)
- **Push-T 的动作对齐提高了接触率。** 固定原视觉 ALT-30 的匹配帧，用真实 T 位姿对齐 8 步动作后，接触数从 106/395 增至 279/395。段内最高覆盖小幅提升，段末覆盖未提高。[配对动作迁移诊断](experiments/pusht_action_transfer/RESULTS.md)
- **Push-T 的相对几何让推杆匹配更准。** 离线相对位置误差从 37.81 降到约 14.5。同边筛选的官方均分为 0.3031，相对旧全局几何的配对区间跨零，闭环中约 72% 的查询需要回退。[规则对比与匹配图](docs/pusht_matching_gallery.md)

## 仓库导航

| 路径 | 内容 |
| --- | --- |
| [experiments/lift_box_dp](experiments/lift_box_dp/README.md) | Lift 对比代码、配置、报告、冻结状态和小型逐局记录 |
| [experiments/alt_robomimic_lift](experiments/alt_robomimic_lift/README.md) | 原始 Lift ALT 实现、数据划分、模拟器验证和动作块实验 |
| [environments/robomimic_lift](environments/robomimic_lift/README.md) | Lift 的 WSL 旧版模拟器配置 |
| [src](src)、[scripts](scripts)、[configs](configs) | Push-T ALT 实现和实验入口 |
| [docs](docs)、[experiments/runs](experiments/runs)、[results](results) | Push-T 审计、运行记录及图表 |

## 复现

Lift-PH 图像数据共有 200 条独立示范，冻结 20 条离线验证和 180 条训练池。[Lift 划分](experiments/alt_robomimic_lift/configs/lift_ph_split42.json) 保存精确轨迹 ID。Lift 对比依赖同级 `DP/diffusion_policy` 参考源码、CUDA Python、既有 HDF5 数据和 WSL Ubuntu 中的 RoboSuite 1.2.0 / MuJoCo 2.0.0。进入 `experiments/lift_box_dp` 后按实验方法文档执行。重训使用新的运行 ID。

Push-T 使用本地 DP 参考项目的 `pusht_cchi_v7_replay.zarr`。数据审计发现 206 条独立轨迹，当前划分可支持 30 和 100 条训练示范。300 和 500 条在当前数据下不可用。运行前核对 [数据审计](docs/data_audit.md)、[来源与版本](docs/upstream_provenance.md) 和 [配置](configs/alt_pusht_base.yaml)。

```powershell
$python = '..\DP\.conda\pusht\python.exe'
& $python -B scripts/run.py audit
& $python -B scripts/run.py prepare-weights
& $python -B scripts/run.py check

$run = 'alt_030_split42_model42_reproduction_v1'
& $python -B scripts/run.py train --run-id $run --scale 30 --model-seed 42
& $python -B scripts/run.py bank --run-id $run
& $python -B scripts/run.py evaluate --run-id $run
```

Git 保存代码、冻结清单、逐局指标与校验和。原始数据、检查点、记忆库、视频和完整控制轨迹留在本地 `artifacts/`。复现既有模型需要对应的本地权重和数据。
