# Push‑T 动作块迁移诊断

## 问题

在检索帧和示范动作完全相同时，用当前 T 块的位姿重新定位动作块，能否使推杆碰到 T 块并把它推向目标？本实验使用仿真真值做动作坐标变换，用于诊断视觉 ALT 的动作复用问题。两种来源分别是已保存的几何真值检索轨迹和视觉 ALT-30 轨迹。

## 固定设置

- 使用原 ALT-30 的 30 条训练示范和冻结的 50 个模拟器种子。每条已存轨迹预先取第 0、40、80、120、160、200、240、280 步的检索快照。提前终止后未到达的快照不计。
- 每个快照沿基线历史动作从头重放。原样回放和变换回放开始时，推杆和 T 的位置、角度及速度必须逐值相等。原样回放逐种子复核原正式运行的最大和终点覆盖分数。
- 固定匹配示范帧及 8 步动作。变换动作先减去示范 T 的原点，再旋转当前与示范的角度差，加上当前 T 的原点。输出仍是绝对推杆目标，超出接口边界时裁剪至 `[0,512]` 并计数。
- 从物理引擎的推杆与 T 接触回调读取首次接触点，记录其在 T 坐标系下最近的外轮廓边。墙壁碰撞不计入。旋转对照只统计两分支都执行 8 步且示范确实转过至少 2° 的样本；分支实际转动也需达到 2°。方向匹配反映与示范动作结果的一致性。
- 比较单段的最大与末步官方覆盖分数，配对差以模拟器种子为簇重采样。两条分支执行后都不接着重查，因此这些分数只衡量相同起点上的局部动作结果。

## 复现

需要原 oracle 和视觉 ALT-30 的本地逐步检索日志。脚本会根据既有 artifact manifest 核对源日志校验和，并验证每个匹配动作来自冻结训练库。用新运行 ID 避免覆盖。

```powershell
$python = 'D:\Code\Course_code\DP\.conda\pusht\python.exe'
& $python -B experiments/pusht_action_transfer/compare_transfer.py --source oracle --run-id oracle_transfer_reproduction
& $python -B experiments/pusht_action_transfer/compare_transfer.py --source visual --run-id visual_transfer_reproduction
```

结果摘要写入 `results/`，两分支的逐段记录留在 `artifacts/`。

[实测结果与正反例](RESULTS.md)
