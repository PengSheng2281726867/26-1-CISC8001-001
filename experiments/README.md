# 实验目录

[按 PPT 页查素材](../docs/presentation/README.md) · [项目总览](../README.md) · [运行登记表](registry.csv)

## 固定更新次数对照

Lift 的 10／30／100 条示范比较固定 800 次更新，使用同一批 24 个验证起点。视觉 Push‑T 的 30／100 条比较固定 1,000 次更新，沿用 50 个正式测试起点。各组用最终权重，训练种子为 42、43、44。[实验方法与复现](matched_budget/EXPERIMENT.md) · [结果](matched_budget/RESULTS.md)

## 全量示范和长训练

Lift 180 条、视觉 Push‑T 186 条。ALT 与 DP 按各自候选检查点在相同验证起点选权重，每个模型运行 50 局正式测试。[实验方法与复现](full_pool/EXPERIMENT.md) · [结果](full_pool/RESULTS.md)

## Lift-PH

模拟抓取、抬升方块。正式测试每组 50 局、每局 1,000 步。轮数验证每个检查点 24 局、每局 250 步。各实验目录中的方法、结果和日志链接到冻结配置及逐局证据。

| 实验 | 已有记录 | 证据入口 |
| --- | --- | --- |
| ALT 原始实现、环境审计与评估 | 30 条示范的视觉与机械臂状态输入，原始闭环轨迹与首次达标诊断 | [方法与复现](alt_robomimic_lift/README.md) · [原始分析](alt_robomimic_lift/docs/final_analysis.md) |
| ALT 动作块长度 | 同一模型与记忆库，比较动作块长度对首次达标的影响 | [动作块消融](alt_robomimic_lift/docs/action_chunk_ablation.md) |
| Lift‑10 | ALT／DP 轮数验证、选模、50 局配对正式测试与补测 | [实验结果](lift_box_dp/experiments/exp_002_lift10_epochs/RESULTS.md) |
| Lift‑30 | ALT／DP 50 局配对正式测试，含示范保持时长审计 | [实验结果](lift_box_dp/experiments/exp_001_lift30/RESULTS.md) |
| Lift‑50 | ALT／DP 多检查点的 24 局轮数验证 | [实验结果](lift_box_dp/experiments/exp_003_lift50_epochs/RESULTS.md) |
| Lift‑100 | ALT／DP 多检查点的 24 局轮数验证及训练更新量对照 | [实验结果](lift_box_dp/experiments/exp_004_lift100_epochs/RESULTS.md) |

[所有规模汇总](lift_box_dp/RESULTS.md) · [对比代码与复现](lift_box_dp/README.md) · [逐轨迹划分](alt_robomimic_lift/configs/lift_ph_split42.json) · [逐局及训练证据](lift_box_dp/evidence/) · [仿真环境](../environments/robomimic_lift/README.md)

## Push‑T｜视觉策略

策略输入为 RGB 与推杆坐标。比较共享冻结训练轨迹和评估种子。每局最多 300 步，执行 8 步后重新决策。最高覆盖分、末步分和成功局数分别记录。

| 实验 | 已有记录 | 证据入口 |
| --- | --- | --- |
| ALT‑30 与 ALT‑100 | 视觉训练、建库、闭环验证、逐帧诊断 | [数据与方法](../docs/pusht_results.md) · [ALT‑30](runs/alt_030_split42_model42_pilot_v1/report.md) · [ALT‑100](runs/alt_100_split42_model42_pilot_v1/report.md) · [全部基线运行](runs/) |
| B1 位姿辅助监督 | 训练时预测 T 位姿、部署时按视觉融合表示检索 | [方法与实测](fixed_alt_encoder/README.md) · [结果解释](fixed_alt_encoder/docs/interpretation.md) |
| B2-B 正样本可行性 | 联合几何相似阈值下的跨轨迹候选覆盖，仅离线分析 | [分析报告](geometry_sampling_analysis/analysis_report.md) |
| DP‑30 与 DP‑100 | 两种规模同轨迹对照，5,000 次更新，独立验证与正式测试 | [方法](pusht_dp/README.md) · [结果](pusht_dp/RESULTS.md) · [逐局证据](pusht_dp/evidence/) |
| DP 长预算与公开权重复评 | 100／186 条、15,000 次更新，以及官方视觉 CNN 同种子复评 | [长训练报告](pusht_dp/LONG_TRAINING.md) · [训练和逐局证据](pusht_dp/evidence_long/) |

[数据审计](../docs/data_audit.md) · [冻结示范 ID 与测试种子](splits/pusht_v7_split42.json) · [当前帧与检索帧图集](../docs/pusht_matching_gallery.md)

## Push‑T｜真值几何与动作迁移

下列几何规则、对齐或学习修正环节使用仿真真值 T 位姿。闭环实验、同一状态分支的单动作块实验，以及离线匹配统计分别记录。

| 实验 | 已有记录 | 证据入口 |
| --- | --- | --- |
| 全局几何真值检索 | 用位置、姿态做规则检索，对比动作执行长度 | [几何检索](geometry_oracle_alt/analysis_report.md) · [动作长度消融](geometry_oracle_alt/results/action_horizon_ablation_20261023T070258Z/report.md) |
| 相对几何与接触侧规则 | 更换检索距离、同边筛选与回退规则，记录多个实际候选 | [规则结果](pusht_contact_matching/RESULTS.md) · [匹配图和配置](pusht_contact_matching/README.md) |
| 固定候选的动作块迁移 | 同起点下原样回放与坐标对齐，检查接触、转向和覆盖进展 | [方法](pusht_action_transfer/README.md) · [结果与正反例](pusht_action_transfer/RESULTS.md) |
| 第一组残差实验 | 全局／相对检索、对齐、无界／有界残差，450 局复核 | [三轮综合报告](pusht_residual/experiments/residual_study/REPORT.md) · [小型证据](pusht_residual/experiments/residual_study/evidence/) |
| 第二组残差实验 | 训练轮数、监督规模、动作块内衰减，96 局验证及 200 局正式测试 | [方法](pusht_residual/experiments/network_study/EXPERIMENT.md) · [结果](pusht_residual/experiments/network_study/RESULTS.md) · [小型证据](pusht_residual/experiments/network_study/evidence/) |

## 参考资料

[原 ALT 数据样例](original_alt_dataset_preview/README.md) · [上游版本与许可](../docs/upstream_provenance.md) · [项目实验记录](../docs/experiment_log.md) · [失败记录](../docs/failure_log.md)
