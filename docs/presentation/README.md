# ALT 与 Diffusion Policy 汇报素材

**主问题：少量示范下，什么时候可以直接复用历史动作，什么时候需要把动作调整到当前场景？** 先用 Lift 和 Push‑T 对比两种策略，再用 Push‑T 的匹配失败引出坐标对齐与残差修正。

[按实验查模型架构图（含 ImageGen 宽屏版）](ARCHITECTURES.md) · [两段仿真短片](MEDIA.md)

## 正文 9 页

| 页 | 标题 | 素材与讲法 |
| ---: | --- | --- |
| 1 | DP 在少量示范下会记住什么？ | 用 He 等人的 Fig. 4 展示推理动作和训练轨迹的相似度。说明这是作者在特定条件下观察到的现象，也是 ALT 的动机。将自己的问题落在两种策略在不同任务中的实测表现。[原论文 §3.4](https://arxiv.org/html/2505.05787v1#S3.SS4) |
| 2 | 生成动作与检索动作 | 用两条流程解释 DP 的条件去噪和 ALT 的编码、最近邻检索、动作回放。两张仿真截图交代任务。说明每个直接对照共享训练轨迹和测试起点，并在各任务页注明控制协议。[策略流程 PNG](materials/methods/alt_vs_dp.png) · [Lift 截图](materials/lift/lift_video_still.png) · [Push‑T 截图](materials/pusht_geometry/pusht_video_still.png) |
| 3 | Lift：抬起来、撑一秒、结束时仍在 | 比较 10、30、100 条。前两组是正式测试，100 条取同为第 40 轮的验证结果。放录像展示“曾达标、末步未达标”。[Lift ALT 架构](materials/architecture_imagegen/lift_alt.png) · [Lift DP 架构](materials/architecture_imagegen/lift_dp.png) · [主图 PNG](materials/lift/lift_outcomes.png) · [表格 CSV](materials/lift/lift_outcomes.csv) · [Lift 短片](MEDIA.md#lift-alt30) |
| 4 | 视觉 Push‑T：能推多近，最后剩多少 | 比较 30、100 条下的视觉 ALT 与 DP，然后突出同为 100 条时 DP 延长训练后的变化。主图的两组指标均为 50 局均值。[视觉 ALT 架构](materials/architecture_imagegen/pusht_alt.png) · [视觉 DP 架构](materials/architecture_imagegen/pusht_dp.png) · [主图 PNG](materials/pusht_visual/visual_scores.png) · [表格 CSV](materials/pusht_visual/visual_scores.csv) · [视觉对照](../../experiments/pusht_dp/RESULTS.md) |
| 5 | 图像匹配的最近帧为什么会推错？ | 主讲 ALT‑30 种子 100000、第 296 步。匹配到示范 67 第 30 帧，余弦相似度 0.9924。请观众观察灰色 T 的朝向和蓝色推杆相对 T 的位置。引出“候选推法是否合适”和“动作能否搬过去”两个问题。[匹配图](../../results/visualizations/test_frame_matches_v1_alt30.png) · [帧来源](../../results/visualizations/test_frame_matches_v1.json) · [图集](../pusht_matching_gallery.md) |
| 6 | 检索 → 坐标对齐 → 残差修正 | 示意从历史 T 横梁一侧开始的动作怎样旋转、平移到当前 T。小 MLP 读几何差和历史 8 步动作，预测逐步修正量。醒目标注**仿真真值 T 位姿**。可用一句话交代 DP 和 Flow Matching 带来的启发，实测模块称「残差回归」。[动作示意 PNG](materials/methods/action_transfer.png) · [规则检索流程](materials/architecture_imagegen/pusht_geometry.png) · [残差网络架构](materials/architecture_imagegen/pusht_residual.png) · [动作块迁移实验](../../experiments/pusht_action_transfer/RESULTS.md) · [Flow Matching](https://arxiv.org/abs/2210.02747) |
| 7 | 哪一步带来最大改善？ | 固定相对几何检索规则，对比原样回放和坐标对齐。先看整局统计，再放 Push‑T 短片，展示一个接近目标后覆盖回落的回合。[主图 PNG](materials/pusht_geometry/alignment.png) · [表格 CSV](materials/pusht_geometry/alignment.csv) · [Push‑T 短片](MEDIA.md#push-t-aligned) |
| 8 | 小网络有没有额外贡献？ | 对比纯对齐、普通残差、衰减残差。把最高分、末步分和成功局数一起读，指出衰减结构提高了普通残差的闭环得分。[主图 PNG](materials/pusht_geometry/residual.png) · [表格 CSV](materials/pusht_geometry/residual.csv) · [末步配对区间](materials/pusht_geometry/final_score_pairing.json) |
| 9 | 结论与下一步 | 用下面三组结论收束。下一步从 RGB 估计 T 位姿，让纯对齐和残差组共用估计值，再检验有视觉误差时残差是否有增益。 |

流程图和统计图同时提供 PNG 和同名 SVG。若要在 PPT 重绘或改中文标签，可直接使用配套 CSV。两张任务截图来自演示录像，Push‑T 截图是渲染画面。

## 三组实验结论

### Lift｜示范规模与训练阶段

| 示范 | ALT／DP 轮数 | 评估 | 曾抬升 ALT／DP | 连续达标 ≥1 秒 ALT／DP | 末步达标 ALT／DP |
| ---: | --- | --- | --- | --- | --- |
| 10 | 10／200 | 正式 50 局，1,000 步 | 19/50 · 24/50 | 18/50 · 23/50 | 0/50 · 0/50 |
| 30 | 40／200 | 正式 50 局，1,000 步 | 36/50 · 49/50 | 34/50 · 49/50 | 1/50 · 7/50 |
| 100 | 40／40 | 验证 24 局，250 步 | 10/24 · 24/24 | 8/24 · 24/24 | 1/24 · 2/24 |

Lift 达标线为方块高于桌面基准 4 厘米。一秒保持是连续 20 步高于阈值。10 条时 DP 点估计较高，配对区间跨零。30 条时 DP 三项领先，ALT 使用 1,800 次更新与约 1,135 万编码器参数，DP 使用 4,600 次更新与约 8,735 万策略参数。参数统计分别对应编码器和策略，ALT 的记忆库另占空间。100 条在约 770 次更新处的保持成功数为 ALT 17/24、DP 12/24，第 40 轮排序反过来。检查点、批次大小和评估时长都需随结果标注。[Lift 总览与逐轮结果](../../experiments/lift_box_dp/RESULTS.md) · [ALT‑30 训练元数据](../../experiments/alt_robomimic_lift/experiments/runs/alt_lift_ph30_image_proprio_seed42_v1/run_metadata.json) · [DP‑30 训练元数据](../../experiments/lift_box_dp/evidence/dp30_seed42_v1/training.json)

### 视觉 Push‑T｜检索覆盖与训练预算

30／100 条的视觉 ALT 最高分均值都约 0.274，扩大训练库的收益在这次实验中没有观察到。DP 等规模短预算下的最高分均值为 0.416／0.482。固定 100 条后，长预算模型的最高和末步均值都提高。长预算同时改变了更新上限和学习率日程。186 条与 100 条在长预算下的配对差区间跨零。[视觉对照](../../experiments/pusht_dp/RESULTS.md) · [长预算与官方复评](../../experiments/pusht_dp/LONG_TRAINING.md)

### 真值位姿诊断｜动作对齐与学习修正

| 执行方式 | 最高分均值 | 末步均分 | 成功局数 |
| --- | ---: | ---: | ---: |
| 原样回放 | 0.251 | 0.218 | 1/50 |
| 纯对齐 | **0.926** | 0.572 | 8/50 |
| 对齐＋普通残差 | 0.722 | 0.389 | 1/50 |
| 对齐＋衰减残差 | 0.862 | **0.583** | **11/50** |

同一套相对几何规则下，主要增益来自对象坐标系的动作对齐。衰减残差相对普通残差提高 0.140，配对 95% 区间为 [0.057, 0.222]。衰减相对纯对齐的末步差为 +0.011，区间为 [−0.076, +0.095]。Push‑T 覆盖率严格超过 95% 就会提前结束，所以这里用「末步覆盖」描述结束状态。达标后继续保持的表现尚未测。两组真值实验都读取模拟器的 T 位置和朝向，视觉策略对照只使用 RGB 和推杆位置。[三轮迁移](../../experiments/pusht_residual/experiments/residual_study/REPORT.md) · [三轮网络改进](../../experiments/pusht_residual/experiments/network_study/RESULTS.md)

## 备份页与复现

- **Lift 答疑**：50 条轮数验证、100 条各检查点、一秒保持和末步协议、示范保持审计。[Lift 实验索引](../../experiments/lift_box_dp/README.md)
- **Push‑T 答疑**：ALT 位姿监督、几何采样与不同检索距离、动作块长度、视觉 DP 的 186 条和官方权重、残差网络轮数和监督规模。[完整实验目录](../../experiments/README.md)
- **图表复现**：在仓库根目录执行下面的命令。读取 Git 中已归档的小型证据，其 `source` 字段指向源文件。

  ```powershell
  $python = '..\DP\.conda\pusht\python.exe'
  & $python -B docs/presentation/build_charts.py
  ```

- **视频复现**：见[素材来源与核验](MEDIA.md)。
