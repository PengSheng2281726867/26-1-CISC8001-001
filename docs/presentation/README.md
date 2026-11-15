# ALT 与 Diffusion Policy 汇报素材

**不同示范规模下，ALT 和 DP 在控制要求不同的 Lift‑PH 与 Push‑T 上，闭环表现如何变化？** Lift‑PH 要求抓取、抬升和保持，Push‑T 要求通过接触把 T 块推到指定位置和朝向。主实验在各任务内比较 ALT 和 DP 的达标、保持或覆盖表现，并额外比较训练预算的影响。

Push‑T 的匹配失败案例引出后续探索：历史动作经过坐标对齐和残差修正后，能否更好地适应当前场景？

[按实验查模型架构图（含 ImageGen 宽屏版）](ARCHITECTURES.md) · [两段仿真短片](MEDIA.md)

## 正文 9 页

| 页 | 标题 | 素材与讲法 |
| ---: | --- | --- |
| 1 | DP 在少量示范下会记住什么？ | 用 He 等人的 Fig. 4 展示推理动作和训练轨迹的相似度。说明这是作者在特定条件下观察到的现象，也是 ALT 的动机。将自己的问题落在两种策略在不同任务中的实测表现。[原论文 §3.4](https://arxiv.org/html/2505.05787v1#S3.SS4) |
| 2 | 两种策略如何比较？ | 用流程图解释 DP 的条件去噪和 ALT 的编码、检索、回放。用两张截图交代两任务的控制要求。主对照在任务内比较示范规模，两方法共用示范和评估起点，固定更新数、取最后权重并重复三个种子。[策略流程 PNG](materials/methods/alt_vs_dp.png) · [Lift 截图](materials/lift/lift_video_still.png) · [Push‑T 截图](materials/pusht_geometry/pusht_video_still.png) · [实验设计](../../experiments/matched_budget/EXPERIMENT.md) |
| 3 | Lift：抬起来、撑一秒、结束时仍在 | 主图比较 10、30、100 条，统一训练 800 次更新，每个模型验证 24 局、每局 250 步。放录像展示“曾达标、末步未达标”。[Lift ALT 架构](materials/architecture_imagegen/lift_alt.png) · [Lift DP 架构](materials/architecture_imagegen/lift_dp.png) · [主图 PNG](materials/matched_budget/lift.png) · [表格 CSV](materials/matched_budget/lift.csv) · [Lift 短片](MEDIA.md#lift-alt30) |
| 4 | 视觉 Push‑T：能推多近，最后剩多少 | 主图比较 30、100 条，统一训练 1,000 次更新，每个模型在相同 50 个起点正式测试。用历史对照补充 DP 延长训练后的表现。[视觉 ALT 架构](materials/architecture_imagegen/pusht_alt.png) · [视觉 DP 架构](materials/architecture_imagegen/pusht_dp.png) · [主图 PNG](materials/matched_budget/pusht.png) · [表格 CSV](materials/matched_budget/pusht.csv) · [历史视觉对照](../../experiments/pusht_dp/RESULTS.md) |
| 5 | 图像匹配的最近帧为什么会推错？ | 主讲 ALT‑30 种子 100000、第 296 步。匹配到示范 67 第 30 帧，余弦相似度 0.9924。请观众观察灰色 T 的朝向和蓝色推杆相对 T 的位置。引出“候选推法是否合适”和“动作能否搬过去”两个问题。[匹配图](../../results/visualizations/test_frame_matches_v1_alt30.png) · [帧来源](../../results/visualizations/test_frame_matches_v1.json) · [图集](../pusht_matching_gallery.md) |
| 6 | 检索 → 坐标对齐 → 残差修正 | 示意从历史 T 横梁一侧开始的动作怎样旋转、平移到当前 T。小 MLP 读几何差和历史 8 步动作，预测逐步修正量。醒目标注**仿真真值 T 位姿**。可用一句话交代 DP 和 Flow Matching 带来的启发，实测模块称「残差回归」。[动作示意 PNG](materials/methods/action_transfer.png) · [规则检索流程](materials/architecture_imagegen/pusht_geometry.png) · [残差网络架构](materials/architecture_imagegen/pusht_residual.png) · [动作块迁移实验](../../experiments/pusht_action_transfer/RESULTS.md) · [Flow Matching](https://arxiv.org/abs/2210.02747) |
| 7 | 哪一步带来最大改善？ | 固定相对几何检索规则，对比原样回放和坐标对齐。先看整局统计，再放 Push‑T 短片，展示一个接近目标后覆盖回落的回合。[主图 PNG](materials/pusht_geometry/alignment.png) · [表格 CSV](materials/pusht_geometry/alignment.csv) · [Push‑T 短片](MEDIA.md#push-t-aligned) |
| 8 | 小网络有没有额外贡献？ | 对比纯对齐、普通残差、衰减残差。把最高分、末步分和成功局数一起读，指出衰减结构提高了普通残差的闭环得分。[主图 PNG](materials/pusht_geometry/residual.png) · [表格 CSV](materials/pusht_geometry/residual.csv) · [末步配对区间](materials/pusht_geometry/final_score_pairing.json) |
| 9 | 结论与后续问题 | 短预算下，ALT 在两任务中都有规模收益，增益出现在哪段随任务而变。早期较长的 DP 配方取得过更高成绩。最后总结真值对齐和残差消融，引出从 RGB 估计 T 位姿后的动作迁移问题。[对照结论](../../experiments/matched_budget/RESULTS.md) |

流程图和统计图同时提供 PNG 和同名 SVG。若要在 PPT 重绘或改中文标签，可直接使用配套 CSV。两张任务截图来自演示录像，Push‑T 截图是渲染画面。

## 三组实验结论

### Lift｜固定短预算与三个达标指标

固定 800 次更新时，10／30／100 条的 ALT 一秒保持率为 **18.1%／59.7%／62.5%**，DP 为 **8.3%／11.1%／9.7%**。30 和 100 条的排序在三个种子上复现。ALT 从 10 到 30 条的增益清楚，从 30 到 100 条的增益有较大不确定性。两种策略的末步达标率均低，适合和短时保持分开讲。[800 次更新补测](../../experiments/matched_budget/RESULTS.md#liftph)

### 视觉 Push‑T｜示范规模与训练预算

固定 1,000 次更新时，30／100 条的 ALT 最高分分别为 **0.237／0.321**，DP 为 **0.152／0.151**。两个规模下，ALT 在三个训练种子均领先。早期视觉实验中 DP 以 5,000 次更新得到 0.416／0.482，ALT 在各自 40 轮得约 0.274。固定 100 条后，延长 DP 训练和学习率日程进一步提高了最高分和末步均分。[1,000 次更新补测](../../experiments/matched_budget/RESULTS.md#视觉-pusht) · [早期视觉对照](../../experiments/pusht_dp/RESULTS.md) · [长预算与官方复评](../../experiments/pusht_dp/LONG_TRAINING.md)

### 真值位姿诊断｜动作对齐与学习修正

| 执行方式 | 最高分均值 | 末步均分 | 成功局数 |
| --- | ---: | ---: | ---: |
| 原样回放 | 0.251 | 0.218 | 1/50 |
| 纯对齐 | **0.926** | 0.572 | 8/50 |
| 对齐＋普通残差 | 0.722 | 0.389 | 1/50 |
| 对齐＋衰减残差 | 0.862 | **0.583** | **11/50** |

同一套相对几何规则下，主要增益来自对象坐标系的动作对齐。衰减残差相对普通残差提高 0.140，配对 95% 区间为 [0.057, 0.222]。衰减相对纯对齐的末步差为 +0.011，区间为 [−0.076, +0.095]。Push‑T 覆盖率严格超过 95% 就会提前结束，所以这里用「末步覆盖」描述结束状态。达标后继续保持的表现尚未测。两组真值实验都读取模拟器的 T 位置和朝向，视觉策略对照只使用 RGB 和推杆位置。[三轮迁移](../../experiments/pusht_residual/experiments/residual_study/REPORT.md) · [三轮网络改进](../../experiments/pusht_residual/experiments/network_study/RESULTS.md)

## 备份页与复现

### Lift 的早期检查点与评估协议

| 示范 | 被评估检查点（ALT／DP） | 评估 | 曾抬升 ALT／DP | 连续达标 ≥1 秒 ALT／DP | 末步达标 ALT／DP |
| ---: | --- | --- | --- | --- | --- |
| 10 | 10／200 轮，独立验证选出 | 正式 50 局，1,000 步 | 19/50 · 24/50 | 18/50 · 23/50 | 0/50 · 0/50 |
| 30 | 40／200 轮，预设最终轮 | 正式 50 局，1,000 步 | 36/50 · 49/50 | 34/50 · 49/50 | 1/50 · 7/50 |
| 100 | 40／40 轮，同轮数切面 | 验证 24 局，250 步 | 10/24 · 24/24 | 8/24 · 24/24 | 1/24 · 2/24 |

Lift 达标线为方块高于桌面基准 4 厘米。一秒保持是连续 20 步高于阈值。表里的轮数是权重所在的**训练 epoch**，每局执行步数列在「评估」中。10 条从各自已测检查点中用独立验证选出权重。30 条按预设预算取最终权重。10 条时 DP 点估计较高，配对区间跨零。30 条时 DP 三项领先，ALT 使用 1,800 次更新与约 1,135 万编码器参数，DP 使用 4,600 次更新与约 8,735 万策略参数。ALT 的记忆库另占空间。[Lift 总览与逐轮结果](../../experiments/lift_box_dp/RESULTS.md) · [ALT‑30 训练元数据](../../experiments/alt_robomimic_lift/experiments/runs/alt_lift_ph30_image_proprio_seed42_v1/run_metadata.json) · [DP‑30 训练元数据](../../experiments/lift_box_dp/evidence/dp30_seed42_v1/training.json)

**训练预算单独比较。** 100 条组两者都取第 40 轮时，ALT 和 DP 分别进行了 6,080 和 3,080 次参数更新。取更新次数接近的检查点时，ALT 第 5 轮（760 次）保持成功 17/24，DP 第 10 轮（770 次）为 12/24。两个切面都使用同一批 24 个验证起点。轮数和更新次数分别衡量数据遍历与优化次数，批大小、训练目标及每次更新耗时也有差异。[Lift‑100 检查点比较](../../experiments/lift_box_dp/experiments/exp_004_lift100_epochs/RESULTS.md)

[Lift 早期图表 PNG](materials/lift/lift_outcomes.png) · [Lift 早期图表 CSV](materials/lift/lift_outcomes.csv) · [Push‑T 早期图表 PNG](materials/pusht_visual/visual_scores.png) · [Push‑T 早期图表 CSV](materials/pusht_visual/visual_scores.csv)

- **Lift 答疑**：50 条轮数验证、100 条各检查点、一秒保持和末步协议、示范保持审计。[Lift 实验索引](../../experiments/lift_box_dp/README.md)
- **Push‑T 答疑**：ALT 位姿监督、几何采样与不同检索距离、动作块长度、视觉 DP 的 186 条和官方权重、残差网络轮数和监督规模。[完整实验目录](../../experiments/README.md)
- **图表复现**：在仓库根目录执行下面的命令。读取 Git 中已归档的小型证据，其 `source` 字段指向源文件。

  ```powershell
  $python = '..\DP\.conda\pusht\python.exe'
  & $python -B docs/presentation/build_charts.py
  ```

- **视频复现**：见[素材来源与核验](MEDIA.md)。
