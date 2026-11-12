# Lift-PH 100 条示范下的 ALT 与 DP

## 研究问题

用相同的 100 条示范训练时，ALT 和 DP 的闭环抬升表现如何？与 50 条实验相比，增加示范后，两种策略在共同检查点上的表现是否提高？保留第 40 轮直接对照，同时检查更早和更晚的训练阶段。

## 数据、模型与训练

使用冻结清单中的 100 条嵌套示范，包含此前的 50 条，和 20 条离线验证轨迹互斥。两种策略都只读当前两路 RGB 图像和当前 9 维本体状态。每次执行原始 7 维增量控制指令，下一步重新决策。

ALT 使用共享、微调的 ImageNet ResNet-18 和本体状态编码器，将当前观测映射到 128 维单位向量。训练使用同帧光度变换构造正样本和 NT-Xent 损失，学习哪些观测适合匹配。第 5、10、40 轮分别重新编码全部训练帧，检索时取余弦最相似的一帧并执行对应动作。优化器及增广沿用 Lift-50。

DP 用两个视觉编码器及当前本体状态作为条件，通过时序 U-Net 预测动作序列中的扩散噪声，保留最终参数的指数移动平均。训练使用 16 步专家动作窗口，推理通过 10 步 DDIM 得到动作序列，每步执行第一条指令。保留第 10、40、150 轮。预热为 175 次参数更新，余弦日程按本次 150 轮定义。50 条实验的 DP 日程是 300 轮，因此同轮数比较还受到学习率轨迹影响。

## 评估与限制

复用两个冻结验证批次，总计 24 个独立初始状态。每个检查点均跑 250 步。主指标是方块连续 20 步超过桌面偏移高度加 4 厘米，同时报告曾达标和终点达标。逐步校验模拟器复位哈希、动作形状和控制范围。旧的 50 个正式测试初始状态保留。

同轮数下，较大的训练集带来更多更新和更大的 ALT 记忆库。两种方法的视觉编码与预训练也不同。每种方法使用一个模型种子。本次比较覆盖预先列出的轮数。

## 复现

使用项目已有的 Windows CUDA Python 和 WSL Lift 模拟器。在本项目根目录运行。复现时使用新的运行 ID。

```powershell
$python = '..\alt_robomimic_lift\artifacts\environments\alt_lift_train\Scripts\python.exe'
& $python -B src/train_alt10.py alt100_epochs_v1 --scale 100
& $python -B src/train.py dp100_epochs_v1 --study --scale 100
& $python -B src/evaluate_alt50.py alt100_epochs_v1 evaluate --method alt --scale 100
& $python -B src/evaluate_alt50.py dp100_epochs_v1 evaluate --method dp --scale 100
& $python -B src/evaluate_alt50.py alt100_epochs_v1 summarize --method alt --scale 100
& $python -B src/evaluate_alt50.py dp100_epochs_v1 summarize --method dp --scale 100
```

固定配置为 [config.json](config.json) 和 [dp_config.json](dp_config.json)。`src/epoch_study.py` 校验嵌套集合及训练配置，`src/evaluate_alt50.py` 复用 24 状态评估和逐步复核逻辑。两份训练入口复用已验证的 ALT 与 DP 架构。
