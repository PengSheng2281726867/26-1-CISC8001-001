# B2-B Geometry Sampling Feasibility Analysis

本目录是独立的数据分析，不包含训练、模型加载、优化器或模拟器调用。不修改 ALT/B1，也不新建划分。所有代码、配置、缓存、输出和报告仅在本目录。

本目录的 `.gitattributes` 禁止Git换行转换，保留各文件被记录SHA-256时的原始字节；旧实验和项目其他目录的属性不变。

## 数据与依赖

- 项目根目录 `experiments/splits/pusht_v7_split42.json` 的 `train_subsets["30"]`，原30条轨迹、3806帧。
- Zarr 路径从根目录 `configs/alt_pusht_base.yaml` 获取，支持修改分析配置的 `project_config` 指向另一个路径配置。
- 不使用通用 PushTData：该类会读取全库状态。本分析直接按30个训练Episode的边界读取 `data/state[start:end, :]`。验证/测试状态不作候选，也不读取动作数组。
- schema与角度证据、确切Episode ID、读取切片及选定状态哈希在 `results/data_audit.json`。冻结全数据指纹仅作既有引用，不重新扫描全库状态；本次实际计算选定数据和Zarr元数据的哈希。
- 现有只读环境：Python3.10、NumPy1.23.5、Zarr2.12.0、PyYAML、Matplotlib；不需要Torch/GPU。Windows使用已有解释器 `D:\Code\Course_code\DP\.conda\pusht\python.exe`，不向其安装依赖。

## 运行与复现

从项目根目录运行：

```powershell
$geoPython = 'D:\Code\Course_code\DP\.conda\pusht\python.exe'
& $geoPython -B -m unittest discover -s experiments/geometry_sampling_analysis/tests -v

# 首次分析；结果已保存时会拒绝覆盖
& $geoPython -B experiments/geometry_sampling_analysis/analyze.py
& $geoPython -B experiments/geometry_sampling_analysis/analyze.py --config experiments/geometry_sampling_analysis/config_absolute.yaml --output-root absolute

# 已有Git结果时，选择新的输出目录进行独立复现
& $geoPython -B experiments/geometry_sampling_analysis/analyze.py --output-root .cache/reproduce_quantile
& $geoPython -B experiments/geometry_sampling_analysis/analyze.py --config experiments/geometry_sampling_analysis/config_absolute.yaml --output-root .cache/reproduce_absolute

# 检查保存的结果；需先生成上面的quantile复现目录
& $geoPython -B experiments/geometry_sampling_analysis/verify_outputs.py
```

`.cache/reproduce_quantile` 已存在时也会拒绝覆盖。可更换 `--output-root`；`verify_outputs.py` 默认用上述quantile目录作逐字节核对。`analyze.py` 可从其他工作目录启动，默认分析配置与输出仍相对于脚本所在目录；指定 `--config` 时路径按当前工作目录解释。输出禁止逃出本目录。

## 统计合同

距离按原世界坐标计算：`dp=L2(p_i-p_j)`、`dT=L2(T_i-T_j)`、`dtheta=abs(wrap(theta_i-theta_j))`。联合距离是用户指定的 `(dp/512)^2+(dT/512)^2+(dtheta/pi)^2`，不是已确定的训练权重。用float64分块计算、float32存内存；不保存/提交任何N×N矩阵。

所有条件排除自身。候选范围为全部其他帧、排除同轨迹±8帧、仅跨轨迹。Top-k按D稳定排序，同距离时沿冻结训练帧顺序决胜。阈值统计同时保留各距离分量、无序/有序帧对数、Anchor失败率和Episode集中程度。

`config.yaml` 为最初预设的分位数网格：由跨轨迹训练帧对的各分量分布校准，近5/10/20分位，远50/75分位。完整57条件×3候选池，不只报告三档示例。观察距离分布发现分位数近阈值偏宽后，增加 `config_absolute.yaml` 的完整物理尺度补充网格；原标准和所有结果保留，修改依据在 `execution_log.md`。

每套配置模拟：3档阈值×3候选池×2唯一性策略×100个结构化Batch；另生成100个完全随机Batch并在9个阈值/候选池组合下计分。这是1800个结构化+100个独立随机Batch，非2700个独立随机实验。两套配置各有2700行计分记录；随机参考帧组在两套配置间也相同。

每个结构化Batch随机无放回选择8个Anchor，每个分配1个Similar和1个Hard（随机均衡4A/4B），最后8个Random。候选均匀选择，不按D选最有利项。唯一性策略拒绝已用帧；另一策略允许跨角色重复。无候选或去重后耗尽则保留缺槽，不补位、不重抽Anchor、不放宽阈值。

随机参考组是均匀无放回32帧，前8个与结构化组共享Anchor；它没有角色配额。其32帧完整率不是几何配额成功率。统计分别保存含缺槽的全部尝试与完整Batch；全批不同帧对数、重复槽位关系数分别记录，不能混淆。

## 输出与检查

- `results/geometry_statistics.json`：距离/时间分布、Top5/10/20、171组条件、Episode端点计数与候选覆盖率。
- `results/batch_sampling_statistics.json`：各组100次模拟汇总，含完成率、缺槽原因、重复帧、Episode覆盖和几何对数。
- `results/batch_details.csv`：2700条逐Batch计分记录。
- `results/primary_anchor_candidate_counts.csv`：全部3806个Anchor在主阈值下的9项候选数，0即失败。
- `results/threshold_sensitivity.csv`：171组完整敏感性统计索引。
- `results/batch_examples.json` / `figure_examples.json`：选择的帧行号及角色实例。行号映射由冻结Episode顺序与Episode内部帧序唯一恢复，不是新划分。
- `absolute/`：另一完整阈值家族的同结构结果/图。
- `results/reproduction_verification.json`：原分位数结果逐字节复算一致，主阈值候选数逐Anchor独立重算，5400行Batch汇总复核。
- `results/slot_feasibility_bounds.json`：原均匀Anchor方案的必要条件概率上界及同100组Anchor的可用性检查；不是最优匹配成功率。
- `results/figure_corrections.json`：只修正CDF非负横轴的旧/新渲染哈希。最初执行记录保留原图版本哈希，最终文件以artifact_manifest为准。
- `figures/` / `absolute/figures/`：距离CDF、近邻来源与误差、覆盖敏感性、Batch比较、固定随机Anchor图例。图例缺候选会直接留空，不为了展示而换Anchor。
- `analysis_report.md`：中文结论、限制、后续建议。几何相似/不同均不等于动作相容/不相容。

10项必要测试覆盖角度周期、分量/联合距离、±8边界、Hard B的OR条件、自排除与并列、配对计数、只读训练切片、缺槽不补、去重失败及确定性。真实数据另做512组独立标量距离复核与24行Top20复核；`verify_outputs.py`对两套主阈值各3806个Anchor的9类关系独立重算并核对所有模拟汇总。

脚本不自动训练B2-B。后续采样/标签建议均未实施。
