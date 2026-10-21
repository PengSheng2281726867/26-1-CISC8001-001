# Geometry-Oracle ALT-30

这是不训练网络的诊断性 Oracle Baseline。输入为仿真器当前真实
`[pusher_x, pusher_y, block_x, block_y, block_theta]`，超出原视觉 ALT 的信息条件。
本实验不加载 ResNet、Embedding 或模型 Checkpoint，不是严格理论性能上界。

已完成一次固定 50-seed 评测：Oracle Mean Score **0.224722502**（原 ALT 0.274423899，B1 0.178184988）。
离线几何误差下降，但闭环没有明确提升；两项配对差异 95% CI 均跨零，重复检索仍然常见。

## 冻结方案

- 数据路径复用 `configs/alt_pusht_base.yaml`；只读读取 Push-T v7 Zarr。
- Split: `experiments/splits/pusht_v7_split42.json`，训练只用 `train_subsets["30"]`。
- 30 条训练 Episode、3,806 帧；20 条原验证 Episode、2,426 帧仅作离线查询。
- 距离：`(pusher_L2/512)^2 + (block_L2/512)^2 + (wrap(angle_delta)/pi)^2`。
- 状态保持原始 float32；距离用 float64；角度按弧度周期距离计算。
- Top-1；距离完全相同按冻结 Episode 顺序、Episode 内帧升序选择。
- 复用 `src/pusht/data.py::PushTData.action_chunk()`，offset=0，8 步原始绝对目标命令，末尾 repeat-last；不跨 Episode。
- 官方 `PushTImageRunner.run` 不修改：50 Seeds 100000–100049，legacy=True，300 步，n_obs_steps=1，n_action_steps=8，n_envs=1；前四个 Seed 保存视频。
- 无动作补偿/插值/混合，无 OOD fallback，无权重搜索。

`config.yaml` 可配置距离权重；**本次正式运行强制 1:1:1**。未来改变权重须建立新实验，不能覆盖此运行目录。

## Oracle 状态同步

独立 `OracleImageEnv` 子类在官方 `_get_obs()` 后捕获只读状态快照，不增加 Observation 字段。
通过官方 AsyncVectorEnv 的 `call('get_attr', 'oracle_snapshot')` 读取当前状态。
Reset 和每个 action chunk 后均检查五维状态与生成观察时的快照完全相等；
每次检索前再次读取，检查没有推进仿真、时间步不变、pusher xy 与最新观察一致。
角度在读取时统一到 `[0,2pi)`，不写入物理对象。

Windows spawn 测试覆盖真实 AsyncVectorEnv；另与原环境在相同 Seed/动作下逐步比较图像、位置、奖励、done 和 info。
No-op 只读接口不改变动力学、Reset、Seed 或 Reward。

## 复现

在仓库根目录 PowerShell 执行；使用已有 Push-T 环境，不安装进上游源码：

```powershell
& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/run.py tests
& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/run.py offline
& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/run.py evaluate
& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/run.py verify
& 'D:/Code/Course_code/DP/.conda/pusht/python.exe' -B experiments/geometry_oracle_alt/run.py report
```

`offline` 和 `evaluate` 使用不覆盖的独立目录；若已有结果，不要删除或覆盖。
在新的干净克隆中执行可完整复现；需要对已有运行重做时，应显式创建新版本目录并记录原因。
`verify` 和 `report` 不运行仿真、无需训练，可对已存在的本地产物重复执行。
`-B` 与运行时环境变量禁止向只读参考仓库写入 pycache；缓存全部指向本目录忽略的 `artifacts/cache/`。

## 验证与统计解释

- 自检索允许完全重复状态返回更早的同状态行；记录自身索引恢复率和零距离率。
- 全部训练帧检查动作索引、值、padding 和边界；正式评测后从完整日志独立重算所有 top-1、距离和动作。
- 离线位置/角度误差与动作 RMSE 单独报告。几何近不保证专家动作相容。
- 原 ALT/B1 对比只读取冻结结果和完整检索日志，不加载或改动旧 Encoder。
- 配对分数 CI：对相同 50 个 Seed 成对 bootstrap，种子 2026、10,000 次、percentile 95% 区间。
- 闭环重复率分母为非首帧的查询转移；同帧且推杆移动 ≤0.001 为描述性停滞指标，不参与策略。
- 旧视觉模型闭环没有保存 T 块真实状态；不虚构其闭环 T 误差。T 误差比较用相同离线验证查询。
- `artifacts/` 中完整 JSONL、视频及 Bank 不上传；提交摘要、50 行分数表、图和报告。Git 不是大文件备份。

## 来源与许可

距离规则由用户指定，本实验实现独立编写。复用本项目的只读数据适配器、动作提取以及官方 Runner 外围观测模式。
DP 源码按现有配置导入、只读使用（MIT，详见仓库 `docs/upstream_provenance.md` 与 `licenses/DP-MIT.txt`）。
不复制整个 DP/ALT 仓库；本实验代码及测试都限于本目录。精确上游 Commit、数据全字节指纹、Split 哈希和代码哈希记录于运行元数据。

最终统计见 [analysis_report.md](analysis_report.md)，执行与修复记录见 [execution_log.md](execution_log.md)。
