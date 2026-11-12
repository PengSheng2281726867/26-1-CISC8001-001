# 10 条示范下，训练到第几轮更好？

## 问题与任务

检验两个问题。其一，在只有 10 条 Lift-PH 图像示范时，训练更久会改善还是损害 ALT 和 DP 的闭环抬升。其二，使用各自验证选出的模型，ALT 能否在这组小样本训练条件下追上 DP。

从上一轮冻结的 30 条列表中按顺序取前 10 条，两个方法使用完全相同的轨迹。20 条离线验证轨迹原样保留。每步输入为同步的两路 84×84 RGB 图像和 9 维本体状态，输出一条原始 7 维 OSC_POSE/Panda 增量动作。夹爪闭合方向、20 Hz、每步重规划与上一轮一致。

## 两条训练曲线

- ALT 沿用共享 ImageNet ResNet-18、两路图像加本体状态融合、128 维单位向量和 NT-Xent 对比损失。第 10、40、100 轮保存同一条训练曲线的检查点。每次用相应编码器把 10 条轨迹的全部当前观测重编码，推理取余弦 top-1 的同帧动作。
- DP 沿用两个图像编码器、条件一维 U-Net、16 步动作目标和 EMA。观察第 40、100、200 轮，训练前固定 200 轮余弦日程及 175 步预热。推理用 10 次 DDIM 去噪，再执行预测动作块的第一步。训练与推理的数值观测统计只从这 10 条拟合。预热步数随样本规模调整，其他超参数继承上一轮。三次快照共享一个优化器和同一学习率轨迹，结论针对这个训练日程。

## 选模与正式评估

预先冻结 8 个独立验证初始状态，编号 90000–90007。每个快照在这 8 个条件下各执行 250 步。先按连续 20 步高于桌面偏移加 4 厘米的局数排序，然后按曾达标局数、终点达标局数排序，仍平局取早的轮数。验证窗口的终点是第 250 步。选出的每种方法仅评估一次此前冻结的 50 个正式初始状态，窗口恢复为 1,000 步。两个阶段均在每步检查同一官方抬升条件，评估过程会从逐步动作记录独立重算全部指标。

训练曲线比较控制了方法内的训练轨迹。10 条与历史 30 条的横向比较还涉及选模规则差异（30 条用最终轮）。小样本子集只有一组，训练种子只有 42。仅用 8 个初始状态选模，容易受单局结果影响。DP 测试按最多 8 局并行，批次推理耗时不可直接与历史 4 局批次比较。

## 入口与复现

在项目根目录，使用已有 Windows CUDA Python 与 WSL Lift 模拟器。若复现，使用新训练 ID。冻结初始状态文件已有时直接复用其哈希。

```powershell
$python = '..\alt_robomimic_lift\artifacts\environments\alt_lift_train\Scripts\python.exe'
& $python -B src/evaluate_epochs.py freeze
& $python -B src/train.py dp10_epochs_v1 --study
& $python -B src/train_alt10.py alt10_epochs_v1
& $python -B src/evaluate_epochs.py validation dp dp10_epochs_v1
& $python -B src/evaluate_epochs.py validation alt alt10_epochs_v1
& $python -B src/evaluate_epochs.py test dp dp10_epochs_v1
& $python -B src/evaluate_epochs.py test alt alt10_epochs_v1
& $python -B src/summarize_epochs.py alt10_epochs_v1 dp10_epochs_v1
```

数据和协议见 [config.json](config.json) 与 `src/epoch_study.py`。训练窗口、模型和保存规则见 `src/train.py`、`src/model.py`、`src/train_alt10.py`。验证与测试入口为 `src/evaluate_epochs.py`，沿用上一轮 `src/evaluate.py` 的闭环执行。

## 2026-11-12｜补测 ALT 早期和 DP 后期检查点

沿用同一批 10 条训练示范、480 帧、同一训练种子、8 个已冻结的验证初始状态、250 步窗口和连续 20 步达标的主要指标。ALT 第 2、5、10 轮各建一次 480 帧记忆库，复核第 10 轮的可重复性。DP 在一次新训练中保存第 150、200、300 轮。其余设置沿用原方案。原 DP 的余弦学习率在 200 轮结束，新曲线的余弦日程持续 300 轮，因此新旧第 200 轮需分别解释。补测比较验证结果，既有 50 局正式测试保持为历史记录。

```powershell
$python = '..\alt_robomimic_lift\artifacts\environments\alt_lift_train\Scripts\python.exe'
& $python -B src/train_alt10.py alt10_supplement_v1 --supplement
& $python -B src/train.py dp10_supplement_v1 --study --supplement
& $python -B src/evaluate_epochs.py validation alt alt10_supplement_v1 --supplement
& $python -B src/evaluate_epochs.py validation dp dp10_supplement_v1 --supplement
```

检查点配置见 [supplement.json](supplement.json)，各次运行的训练损失、更新步数、闭环日志及选模结果保存在对应的 `artifacts/` 子目录。

## 2026-11-12｜新增 16 个验证初始状态

同六个补测检查点各新增 16 局，使用种子 90008–90023，每局仍为 250 步。逐个复位校验观测和模拟器状态的哈希，并检查与原验证状态和正式测试状态无重叠。固定 DP 动作采样种子 46，用于新增验证的各检查点。原 8 局保留种子 44 的结果。新增 16 局和合计 24 局分别汇报，正式测试的 50 局独立保留。

```powershell
$python = '..\alt_robomimic_lift\artifacts\environments\alt_lift_train\Scripts\python.exe'
& $python -B src/extend_epoch_validation.py freeze
& $python -B src/extend_epoch_validation.py alt
& $python -B src/extend_epoch_validation.py dp
& $python -B src/extend_epoch_validation.py summarize
```

冻结文件已存在时直接使用。两个方法分别从既有运行目录加载检查点和记忆库，校验 SHA-256。新增评估写入各运行的 `validation_extra16_e*/`，汇总时逐步复核新增及原验证记录。
