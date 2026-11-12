# 机器人模仿学习实验｜ALT 与 Diffusion Policy

本仓库研究机器人用少量专家示范学习闭环控制时，**直接检索专家动作**与**学习生成动作**各有什么优势。目前有两个模拟任务。Push-T 用图像和推杆位置控制 T 形物块，只研究动作查找表（ALT）。Lift-PH 用两路图像和机械臂本体状态抓取并抬升方块，在相同示范与初始状态下比较 ALT 和扩散策略（DP-CNN）。

## 已完成的实验

| 任务 | 数据与评估 | 主要结果 | 实验记录 |
| --- | --- | --- | --- |
| Lift，30 条示范 | 两种策略各跑 50 局，每局 1,000 步 | 连续抬升至少 1 秒，ALT 34/50，DP 49/50 | [方法与命令](experiments/lift_box_dp/experiments/exp_001_lift30/EXPERIMENT.md) · [结果](experiments/lift_box_dp/experiments/exp_001_lift30/RESULTS.md) |
| Lift，10 条示范 | 先用 8 个独立状态选轮数，再在 50 个状态正式测试 | ALT 第 10 轮 18/50，DP 第 200 轮 23/50 | [方法与命令](experiments/lift_box_dp/experiments/exp_002_lift10_epochs/EXPERIMENT.md) · [结果](experiments/lift_box_dp/experiments/exp_002_lift10_epochs/RESULTS.md) |
| Lift，10 条示范轮数补测 | 六个检查点各跑同一组 24 局，每局 250 步 | ALT 第 5、10 轮均为 9/24。DP 第 150、200、300 轮分别为 9/24、11/24、17/24 | [逐轮汇总](experiments/lift_box_dp/experiments/exp_002_lift10_epochs/validation_extra16_summary.json) |
| Lift，50 条示范 | ALT 四个检查点，各跑同一组 24 局 | 第 5、10、40、100 轮连续保持 1 秒分别为 15/24、16/24、13/24、11/24 | [方法](experiments/lift_box_dp/experiments/exp_003_lift50_epochs/EXPERIMENT.md) · [结果](experiments/lift_box_dp/experiments/exp_003_lift50_epochs/RESULTS.md) |
| Push-T，30 与 100 条示范 | 同一组 50 个状态，官方图像 Runner | ALT 平均官方分数分别为 0.2744、0.2745 | [逐规模汇总](results/scale_summary.csv) · [配对结果](results/paired_scale_comparison.json) |

## 值得关注的结果

- **Lift-30 上 DP 更常完成短时抬升，长期保持仍是难点。** 连续保持 1 秒的成功率差为 30 个百分点，配对 95% 区间为 +16 到 +44 个百分点。到第 1,000 步时，只有 ALT 的 1 局和 DP 的 7 局仍达标。30 条专家轨迹在达标后的记录只有 5 步，约 0.25 秒，方块仍在上升。[Lift-30 结果与示范审计](experiments/lift_box_dp/experiments/exp_001_lift30/RESULTS.md)
- **10 条示范时，两种策略表现更接近。** 50 局正式测试中，DP 保持成功率比 ALT 高 10 个百分点，配对 95% 区间为 −8 到 +28 个百分点。这组实验无法确定少样本下的稳定排序。10 条和 30 条使用了不同的选模规则和更新预算，差距变化也可能受这些因素影响。[Lift-10 正式对比](experiments/lift_box_dp/experiments/exp_002_lift10_epochs/RESULTS.md)
- **训练成本与闭环表现值得一起看。** 在同一组 24 局验证中，ALT 第 5 轮用 75 次更新达到 9/24，与第 10 轮相同。DP 第 150 轮也达到 9/24，继续到第 300 轮达到 17/24。结果来自一组训练子集和一个训练种子。参数更新次数、实际训练时间和预训练条件应分别解读。[轮数补测](experiments/lift_box_dp/experiments/exp_002_lift10_epochs/RESULTS.md)
- **ALT-50 的早期检查点表现更好，训练更久仍出现回落。** 同样是第 10 轮和同一组 24 局，ALT 从 10 条示范时的 9/24 提升到 50 条时的 16/24。50 条下继续到第 100 轮降为 11/24，尽管对比训练损失下降。同轮数的更新次数和记忆库也随数据规模扩大。这是一个训练种子的验证结果。[ALT-50 轮数结果](experiments/lift_box_dp/experiments/exp_003_lift50_epochs/RESULTS.md)
- **Push-T 上扩数据没有带来可确认的分数提升。** ALT-100 相对 ALT-30 的配对差为 +0.0001，95% 区间为 −0.0970 到 +0.0932。两种规模都出现相似图像检索到几何位置不匹配的动作。[检索失败分析](docs/retrieval_failure_analysis.md)

Lift 的「保持」指连续 20 个控制步中方块高度均超过桌面偏移高度加 4 厘米。50 局正式测试和 24 局轮数验证的初始条件及回合长度不同，各自解读。Push-T 使用官方覆盖得分，不能与 Lift 成功率直接对照。

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
