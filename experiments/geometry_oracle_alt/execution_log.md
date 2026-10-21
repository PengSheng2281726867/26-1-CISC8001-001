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
