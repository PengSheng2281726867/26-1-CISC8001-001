# Geometry-Oracle ALT-30 执行记录

## 边界与冻结设置

本轮新任务授权真实五维状态 Oracle 输入；它是独立诊断实验，不改变原 ALT/B1 的输入定义。
用户要求覆盖本实验范围内原 AGENTS 的视觉输入限制；不训练网络，不改变权重或评测协议。
所有新增文件限于本目录，不改根目录实验登记表或旧结果；本文件承担本实验运行/失败/Git 台账。

## 审查

- 阅读根 AGENTS.md；main 分支工作区初始干净，HEAD `aa07b3fe788a50ed854986d9c8bae16307ca3349`，与 origin/main 同步。
- Remote：`https://github.com/PengSheng2281726867/26-1-CISC8001-001.git`。
- 检查原 ALT/B1 的配置、离线诊断、50-seed 分数和完整检索日志；只读取已存在产物。
- 检查 DP 的 PushTEnv、PushTImageEnv、MultiStepWrapper、VideoRecordingWrapper、AsyncVectorEnv 和官方 ImageRunner。
- 状态角度来源是 Pymunk block.angle，单位弧度；Oracle getter 对角度取模而不改变环境对象。
- 环境动作是绝对目标位置，PD 控制；状态在 step 的物理推进后生成观察。数据 action_chunk 延用已验证 offset=0 与 repeat-last。
- 全数据指纹、上游状态、所用代码/配置哈希在每个阶段读取真实文件记录。

## 实现与验证

新增独立几何 Bank、Top-1 Policy、只读状态子类/AsyncVectorEnv 适配、测试和分阶段 CLI。
正式命令先通过测试和离线 gate，且拒绝覆盖现有正式运行目录。
完整异常自动写入 `artifacts/failure_events.jsonl`；实际失败和修复原因会在下方补充。

2026-10-21：`python -B experiments/geometry_oracle_alt/run.py tests` 通过 5 项测试，首次执行无失败。
`offline` 通过：Bank 为 30 Episode / 3,806 帧，源状态逐项一致，全部自检索返回自身且距离零，无重复状态行。
验证集 2,426 帧的均值：pusher=22.958639、T位置=29.173206、角度=8.306185°、8步动作RMSE=36.510392。
与旧 privileged geometry reference 的细小位置差异来自 float64 距离运算；本轮未改规则或按结果调权。
审阅时修正了许可文档链接，并补充依赖版本与计时/硬件元数据；不改变检索或环境行为。
为覆盖该最终代码快照，提交前重新运行测试；离线数值不需要重算，因为数学/采样实现未改变。

## 实现里程碑与正式评测

- 实现 Commit：`dd211d13f2dda2b905c4aa51fe7373f1420d4e38`，消息 `feat(oracle): validate training-free ALT-30 state retrieval`；已成功 Push 到 origin/main。
- 正式评测从该干净 Commit 启动。命令：`python -B experiments/geometry_oracle_alt/run.py evaluate`。
- 正式运行开始于 2026-10-21T11:18:11.930945+00:00，完成于 11:18:48.786226+00:00；Runner 内部计时 31.750787 秒。
- 仅运行一次正式 50-seed 评测，未失败或重试。权重始终为 1:1:1，没有训练、参数搜索或新的划分。
- Mean Score `0.22472250190183934`，50 Seeds、14,703 环境步、1,863 次检索；同步检查全部通过，pusher 最大舍入误差约 1.52e-5。
- 前四个 Seed 视频和完整日志保存在本目录忽略的 `artifacts/geometry_oracle_alt030_v1/`。

## 复核、负面结果诊断与报告

- `python -B experiments/geometry_oracle_alt/run.py verify`：从完整日志重新计算全部 1,863 次 argmin、距离分量、原始状态、动作/索引，50 回合步数以及本地产物哈希，全部通过。
- `python -B experiments/geometry_oracle_alt/run.py report`：从冻结 ALT/B1 CSV 和完整日志计算配对差异、共同离线查询的误差对比与重复率，生成中文报告和图。
- Oracle − 原 ALT：-0.049701，paired bootstrap 95% CI [-0.157576, +0.058016]；Oracle − B1：+0.046538，95% CI [-0.029137, +0.125975]。两项均跨零。
- 真实状态检索没有改善到高分；重复同帧比例 70.22%，41/50 回合以至少三次同帧匹配结束。
- 检查前四个固定 Seed 的视频与检索轨迹：100001/100002/100003 末尾重复同帧 36/35/33 次；非恒定动作块重复，查询时状态可能完全相同。额外静态分镜显示 100001 的推杆在块内移动再回到原位，T 块没有前进。
- 区分实际证据与假设：上述支持“存在重复回放/周期回路”，不能单独证明 8 步长度、未建模速度或局部覆盖不足中的哪一个是唯一原因。Oracle 闭环 T 位置均值误差仍有 64.45，因此没有声称匹配完全准确。
- 正式运行后仅扩展 `reporting.py` 的可复现视频/轨迹诊断和结论文字，增加报告输入/脚本哈希；未修改 Bank、Policy、Runner、权重或正式结果。报告命令成功重新生成；无需重新运行仿真。
- 没有技术失败、权重调整或科学设置修复。本地忽略目录保留所有大产物；Git 只提交小型统计、图表和文档。

## 结果里程碑

结果提交消息：`exp(oracle): report fixed 50-seed ALT-30 state retrieval`。
提交前检查仅本实验目录有变化，原 ALT/B1 和上游参考项目保持不变；结果 Commit SHA 由 Git 历史及最终回复记录，避免在被哈希提交内自引用。
完成结果 Push 后停止，不启动任何网络训练或新实验。
