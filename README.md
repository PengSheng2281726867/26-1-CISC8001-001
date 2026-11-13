# ALT 与 Diffusion Policy｜机器人模仿学习课程项目

当机器人只有少量专家示范时，我们想知道：**从记忆里找一段相似的动作，和训练网络为当前观测生成动作，分别需要多少示范和训练？** 我们用动作查找表（ALT）与扩散策略（Diffusion Policy，DP）在两个模拟任务上做闭环对比。每个直接对照共用专家轨迹与测试初始状态，同时记录训练预算和各自的模型设置。

| 任务 | 研究问题 | 入口 |
| --- | --- | --- |
| Lift-PH 抬升方块 | 少量示范下，能否抬过高度线并保持？增加训练轮数有什么影响？ | [结果总览](experiments/lift_box_dp/RESULTS.md) · [复现索引](experiments/lift_box_dp/README.md) |
| Push-T 推 T 形块 | 从图像与推杆坐标出发，检索动作和生成动作各能把 T 推到哪里？ | [视觉 ALT 与 DP 对照](experiments/pusht_dp/RESULTS.md) · [冻结划分](experiments/splits/pusht_v7_split42.json) |
| Push-T 真值几何诊断 | 若准确知道 T 的位置和朝向，示范动作能否先对齐、再由小网络修正？ | [两组残差实验](experiments/pusht_residual/README.md) · [当前帧与训练候选图集](docs/pusht_matching_gallery.md) |

## Lift-PH｜示范效率和训练效率

正式测试每种设置 50 局，每局最多 1,000 步。「保持至少 1 秒」指连续 20 个控制步都高于桌面基准高度加 4 厘米。终点是否仍达标另算。

| 示范 | ALT 保持 ≥ 1 秒 | DP 保持 ≥ 1 秒 | 检查点（ALT / DP） |
| ---: | ---: | ---: | --- |
| 10 条 | 18/50 | 23/50 | 10 / 200 轮 |
| 30 条 | 34/50 | 49/50 | 40 / 200 轮 |

10 条时差 5 局，配对 95% 区间跨零。30 条时 DP 多成功 15 局，两种规模的选模规则有差异。50 和 100 条另有 24 局轮数验证。在 100 条的验证中，约 770 次更新时 ALT 和 DP 分别为 17/24 和 12/24，第 40 轮分别为 8/24 和 24/24。这说明训练阶段能改变方法排序。DP-100 第 150 轮虽有 24/24 达到一秒保持，终点达标仅 6/24。短时抬升和持续保持应分别衡量。[逐轮结果与训练口径](experiments/lift_box_dp/RESULTS.md)

## Push-T｜先比较视觉策略，再拆解检索失败

视觉 ALT 用一张 RGB 图像与推杆二维坐标，在训练示范中做最近邻检索。视觉 DP 用最近两帧的同类输入，通过条件扩散生成动作。每次都执行最多 8 步绝对推杆目标，正式评估共 50 局，每局最多 300 步。主分数是每局最高归一化目标覆盖分的均值，末步覆盖和成功局数另报。

| 示范 | 视觉 ALT | 视觉 DP | DP 减 ALT（配对 95% 区间） |
| ---: | ---: | ---: | --- |
| 30 条 | 0.2744 | **0.4156** | +0.1412［+0.0466，+0.2359］ |
| 100 条 | 0.2745 | **0.4820** | +0.2074［+0.1252，+0.2896］ |

DP 用 30 条的均分已高于 ALT 用 100 条。各 DP 规模训练 5,000 次更新，从 30 增至 100 条时均分多 0.0664，配对区间跨零。完整成功的局数仍少，DP 的 30 和 100 条组分别为 1/50 和 0/50。这里观察到的是覆盖得分的优势。DP 使用较窄的条件 U-Net 与 16 步 DDIM，两种方法的观测历史、批大小与选模规则有差异。[逐局比较与训练预算](experiments/pusht_dp/RESULTS.md)

视觉 ALT 的匹配图显示，两张图看起来相似时，推杆也可能位于 T 的另一侧。我们试过给图像分支加 T 位姿监督。验证帧的平均 T 角差从 38.3° 降到 17.3°，推杆误差却变大，闭环均分也降到 0.178。离线几何阈值分析发现，30 条示范中只有 27.8% 的锚帧能找到跨轨迹、三项误差都达标的候选。这些结果把困难落到了三个具体量上：推杆相对 T 的位置、T 的桌面位置与朝向，以及训练库是否有可迁移的动作。[视觉适配与逐项诊断](docs/pusht_results.md) · [匹配图集](docs/pusht_matching_gallery.md)

**延长 DP 训练后的补测。** 固定 100 条示范，更新上限从 5,000 增到 15,000 并拉长学习率日程，正式最高覆盖均分从 0.4820 升到 **0.7835**。同预算用全部 186 条训练轨迹得 **0.8440**，末步均分 **0.7175**，成功 **15/50**。186 条相对 100 条的配对区间跨零。上游官方视觉 CNN 检查点在同一测试种子复评为 0.8707，保留其原有训练划分与模型配置。[长预算结果与官方复评](experiments/pusht_dp/LONG_TRAINING.md)

### 如果已知 T 的位姿，历史动作能复用到什么程度？

这部分把仿真器提供的 **T 真值位置和朝向**用于规则检索与动作对齐。它是对检索和动作迁移的诊断。视觉 ALT 与 DP 的主对照仅使用图像和推杆位置，真实 T 位姿应由视觉系统估计。

1. **规则匹配。** 用 T 坐标系里的推杆距离找历史动作，并兼顾 T 的位置和朝向。相对位置更匹配，原样回放仍只有约 0.251。
2. **坐标对齐。** 把同一候选动作先表达在历史 T 的坐标系里，再按当前 T 的位姿转回桌面。这样即使 T 转过，示范里推横梁的动作仍会移到当前横梁上。固定相对检索规则，对齐把 50 局均分从 **0.251 提到 0.926**。
3. **小网络修正。** 让网络学专家动作和对齐动作之间的差，尝试从最近示范迁移到当前状态。第一组实验中无界残差得 0.322，限制修正幅度后升到 0.618。第二组固定 30 条检索库，比较训练时长、增加监督示范和块内衰减。衰减残差比同预算普通残差从 **0.722 提到 0.862**，仍低于纯对齐的主分数。增加监督示范到 100 条、保持 435 次更新时得 0.624。这里可以看到修正网络有改进空间，也能看到离线动作更接近专家时，闭环分数仍可能下降。[坐标对齐与残差](experiments/pusht_residual/experiments/residual_study/REPORT.md) · [网络三轮迭代](experiments/pusht_residual/experiments/network_study/RESULTS.md)

## 结构与复现

| 内容 | 路径 |
| --- | --- |
| Lift ALT 与 DP 的代码、训练轮数验证和正式成绩 | [experiments/lift_box_dp](experiments/lift_box_dp/README.md) |
| Lift 原始 ALT、数据划分与模拟器 | [experiments/alt_robomimic_lift](experiments/alt_robomimic_lift/README.md) · [environments/robomimic_lift](environments/robomimic_lift/README.md) |
| Push-T ALT、数据审计、原始运行记录 | [src](src) · [scripts](scripts) · [数据审计](docs/data_audit.md) · [运行记录](experiments/runs) |
| Push-T DP 的训练、配对统计与证据 | [experiments/pusht_dp](experiments/pusht_dp/README.md) |
| Push-T 位姿真值诊断、对齐与动作修正 | [experiments/pusht_residual](experiments/pusht_residual/README.md) · [匹配图集](docs/pusht_matching_gallery.md) |

Push-T v7 共 206 条独立示范，预留 20 条后，原计划的 30 和 100 条可对照，300 和 500 条超过当前训练池容量。项目记录了精确示范 ID、种子、训练步数和逐局成绩。运行依赖本地数据与 [参考源码](docs/upstream_provenance.md)，路径见 [基础配置](configs/alt_pusht_base.yaml)。从根目录运行的示例：

```powershell
$pushTPython = '..\DP\.conda\pusht\python.exe'
& $pushTPython -B scripts/run.py audit
& $pushTPython -B scripts/run.py prepare-weights
& $pushTPython -B scripts/run.py check

$altRun = 'alt_030_split42_model42_reproduction'
& $pushTPython -B scripts/run.py train --run-id $altRun --scale 30 --model-seed 42
& $pushTPython -B scripts/run.py bank --run-id $altRun
& $pushTPython -B scripts/run.py evaluate --run-id $altRun

& $pushTPython -B experiments/pusht_dp/run.py study --run-id split42_model42_reproduction
```

Lift 的 HDF5 数据、WSL 模拟器和执行命令见 [Lift 复现索引](experiments/lift_box_dp/README.md)。Git 保存小型结果与配置。原始数据、检查点、动作库和视频保存在本地。两项任务的正式模型对照目前均使用单训练种子，评估种子上的配对区间衡量的是固定模型的初始状态波动。
