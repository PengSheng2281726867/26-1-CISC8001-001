# Lift 视觉检索与单步动作修正

本实验固定 30 条示范训练的双相机 ALT 编码器和余弦检索记忆库。原 ALT 直接回放最相似帧的动作。实验组仅训练一个小型 MLP，预测平移和旋转的六维修正量。两组均按 20 Hz 每步重新检索，夹爪命令取自被检索帧。图像和机器人本体状态是唯一在线观测。

首轮正式测试中，连续保持至少一秒由原 ALT 的 34/50 降到 19/50；留出示范上的动作误差下降。见 [首轮实验报告与逐局证据](results/lift30_residual_v1/REPORT.md)。

100 条实验同时扩大编码器训练集、记忆库和残差网络训练集。沿用已有 ALT-100 第 40 轮权重与同轮记忆库；每个查询只从其他 99 条训练轨迹检索。MLP 的结构、训练轮数、验证选择和正式评测起点沿用 30 条实验。两种规模按相同轮数训练，每轮更新次数随样本数变化。100 条实验重新执行纯 ALT 和 ALT＋MLP 两组正式测试。复现时为下列 `train`、`evaluate` 和离线诊断命令加入 `--alt100-source ../lift-box-dp/artifacts/alt100_epochs_v1`，并使用独立运行名。

## 训练样本与网络

阶段诊断见 `phase_study.py`。它用相同的 ALT-100 视觉排名比较直接回放与前五名同阶段筛选。历史阶段由保存的示范状态恢复，当前阶段由仿真器实时判断。使用官方高度阈值与 RoboSuite 双指垫接触检查。该实验测量真实阶段信息的作用。

物体相对目标实验见 `waypoint_study.py`。它在整个同阶段记忆库内检索，比较直接回放、几何对齐和几何对齐加可学习修正。三组使用仿真器阶段，后两组额外使用真值方块位置。这是独立的真值诊断。

```powershell
& $liftPython -B experiments/alt_robomimic_lift/residual/waypoint_study.py lift100_waypoint_repeat --alt100-source ../lift-box-dp/artifacts/alt100_epochs_v1
```

执行前需要已有阶段诊断的本地 `artifacts/residual/lift100_phase_v1/bank_phases.json`。该标签文件可用 `phase_study.py` 重新生成。添加 `--smoke` 并使用独立运行名可做短程验证。

视觉定位消融见 `visual_geometry.py`。轻量共享 CNN 从两张 RGB 图像提取空间关键点和全局特征，与机器人本体状态一起回归「方块中心减末端位置」的三维向量。使用同一组 100 条示范训练，20 条独立留出示范用于选检查点。实时查询与历史记忆均由定位网络估计位置，阶段仍来自仿真器。正式结果与本地 `lift100_waypoint_v1` 两组对照配对。

完整的在线视觉与本体状态输入实验见 `visual_phase.py`。它冻结既有定位权重，训练独立阶段分类器，使用当前相机图像和机器人本体状态估计阶段。训练示范的标签预先写入检索库。在线接口只提供两路 RGB、末端位置与朝向、手指位置。复现命令如下，需要先完成上述视觉定位和阶段标注实验。

```powershell
& $liftPython -B experiments/alt_robomimic_lift/residual/visual_phase.py lift100_visual_phase_repeat --alt100-source ../lift-box-dp/artifacts/alt100_epochs_v1
```

```powershell
& $liftPython -B experiments/alt_robomimic_lift/residual/visual_geometry.py lift100_visual_geometry_repeat --alt100-source ../lift-box-dp/artifacts/alt100_epochs_v1
```

100 条示范上的输入消融保持 ALT 的视觉检索。训练时给 `train` 增加 `--no-embedding` 和 `--alt100-source`，使用新的运行目录。从头训练的 15→64→64→6 映射只接收 8 维相对本体状态和 7 维历史命令。评测和诊断从训练记录读取输入宽度，使用同一 ALT-100 来源。消融汇总会读取并校验既有 `lift100_residual_v1` 的逐局结果。

每个训练帧作为一次查询。检索时屏蔽同一整条轨迹，在另外 29 条轨迹中找余弦最近邻。两份动作都取自各自查询时刻的 `actions[t]`。监督标签是当前专家命令前六维减去检索命令前六维，单位为归一化 OSC 控制命令。

输入含 128 维融合特征差、3 维世界坐标位置差、3 维相对旋转向量、2 维手指位置差和 7 维历史命令。旋转由 `R_query × R_neighbor⁻¹` 得到。各输入列仅用训练对拟合标准化参数。两层宽度 64 的 ReLU 网络输出六维残差，末层权重与偏置从零开始。执行时将残差加回历史命令，对前六维限制到控制器范围，原样传递第七维夹爪指令。

20 条独立留出示范供离线诊断。训练阶段每个检查点报告的 RMSE 是六维残差预测相对监督标签的误差，计算于执行裁剪之前。选模使用另行冻结的 8 个仿真起点，各跑 250 步，在第 5、20、40 轮之间首先比较连续保持至少 20 步的局数，再比较曾经达标、末步达标和较早轮数。正式测试使用预先保存的 50 个测试起点，每局 1,000 步。对照组复用相同模型、记忆库、初始化和控制协议下已有的逐局结果。

## 复现

从仓库根目录在 PowerShell 执行。训练环境需要 CUDA PyTorch、TorchVision、SciPy 和 h5py。仿真需要仓库原有的 WSL Ubuntu Lift 环境。完整训练和评测依赖本地原始 HDF5、冻结 ALT 权重与记忆库。每次训练取新的运行名。

```powershell
$liftPython = '../DP/.conda/pusht/python.exe'
$study = 'experiments/alt_robomimic_lift/residual/study.py'
& $liftPython -B experiments/alt_robomimic_lift/residual/test_study.py
& $liftPython -B $study train lift30_residual_repeat
& $liftPython -B $study evaluate lift30_residual_repeat
& $liftPython -B $study summarize lift30_residual_repeat
& $liftPython -B experiments/alt_robomimic_lift/residual/diagnose_actions.py lift30_residual_repeat
```

训练检查点、近邻索引、逐步动作和奖励保存在 Lift 目录的 `artifacts/residual/<run_id>/`。汇总命令会核对逐步动作与成功记录，并写出轻量化逐局证据与配对差值。最后一条命令分开计算平移与旋转误差，并统计夹爪命令与专家不一致的帧数。协议细节见 [protocol.json](protocol.json)。
