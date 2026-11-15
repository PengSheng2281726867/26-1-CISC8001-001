# 固定更新次数补测记录

## 2026-11-15

- 冻结同任务内的示范规模、更新预算、训练种子、评估起点和最终检查点规则。Lift 选择靠近已有 760／770 次更新结果的 800 次更新切面。Push‑T 选择原 DP 检查点日程中已有的 1,000 次作为短预算对照，并按该预算执行完整学习率日程。
- 确认 GitHub 基线为 `014f0cbbe8c5ae0a07fb4fd7fdec92a1e6ea9432`。训练沿用本地 DP Python 环境与 NVIDIA GeForce RTX 5090，Lift 闭环使用 Ubuntu WSL 下的旧版 RoboMimic／RoboSuite 仿真器。
- 两项任务各对 ALT 与 DP 执行两次训练更新的冒烟验证。Lift 每组重放 2 个冻结起点、每局 25 步。Push‑T 的 ALT 在 2 个测试起点检查官方 Runner，DP 在 1 个验证起点检查官方 Runner 的训练后评估。冒烟数据独立保存。
- Push‑T 启动时使用[早期配置快照](protocol_before_lift_fixtures.json)。随后在[主配置](protocol.json)中写明 Lift 的验证起点范围等仿真参数。导出时复核两个配置的 Push‑T 条件与公共参数一致。
- Lift ALT 的评估队列执行 `run_lift.py alt` 时，遇到同名 Lift‑30 种子 42 的并行预训练尚未写完。训练入口的独占目录创建返回 `FileExistsError`，未覆盖产物。等预训练完成后重启评估队列，校验该权重的配置、步数和校验和，再执行 24 局评估。训练与评估条件保持一致。
- 18 个 Lift 条件和 12 个视觉 Push‑T 条件均完成。导出逐局成绩和逐模型均值，核对 432 局 Lift 验证的动作轨迹以及 600 局 Push‑T 正式测试的得分汇总。检查评估种子、示范 ID、预算和权重规则。项目原有 7 项测试通过。[归档证据](evidence/) · [结果分析](RESULTS.md)
