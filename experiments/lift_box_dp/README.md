# Lift 任务上的扩散策略与动作检索

本项目研究 Panda 机械臂如何把桌面方块抬高，以及抬高后能否保持。扩散策略（Diffusion Policy，DP）学习在当前观测下生成一段增量控制动作。动作查找表（Action Lookup Table，ALT）根据当前观测检索专家帧并回放对应动作。

| 实验 | 目标 | 状态 |
| --- | --- | --- |
| [exp_001_lift30](experiments/exp_001_lift30/EXPERIMENT.md) | 冻结 30 条示范、同条件比较 DP-CNN 与 ALT | 已验证，见 [RESULTS.md](experiments/exp_001_lift30/RESULTS.md) |
| [exp_002_lift10_epochs](experiments/exp_002_lift10_epochs/EXPERIMENT.md) | 用同一批 10 条示范比较训练轮数，并评估选定检查点 | 已验证，见 [RESULTS.md](experiments/exp_002_lift10_epochs/RESULTS.md) |
| [exp_003_lift50_epochs](experiments/exp_003_lift50_epochs/EXPERIMENT.md) | 用相同的 50 条示范比较 ALT 与 DP 的训练轮数 | 已验证，见 [RESULTS.md](experiments/exp_003_lift50_epochs/RESULTS.md) |
| [exp_004_lift100_epochs](experiments/exp_004_lift100_epochs/EXPERIMENT.md) | 相同 100 条示范与 24 个验证状态下比较 ALT 与 DP | 已验证，见 [RESULTS.md](experiments/exp_004_lift100_epochs/RESULTS.md) |

从本目录运行。代码复用同仓库的 `../alt_robomimic_lift` 和仓库同级的 `DP/diffusion_policy`，并使用已验证的 ALT-Lift 数据、权重与模拟器。训练和推理使用现有 Windows CUDA Python，环境执行使用现有 WSL Ubuntu 的 RoboSuite 1.2.0 / MuJoCo 2.0.0。详细配置与命令见实验文档。生成的检查点和原始逐步日志保存在本目录的 `artifacts/`。

在固定的 50 对初始条件下，DP-CNN-30 有 49 次曾抬起并连续保持至少 1 秒，ALT-30 分别是 36 次曾抬起和 34 次保持。两种方法到回合终点都很少维持抬升。当前结果来自单个训练种子，详见实验报告。

[evidence/](evidence/) 保存原始运行配置、训练曲线、逐局成绩和检查点校验和，供 GitHub 上核验报告。实际运行从本目录的 `artifacts/` 加载检查点、记忆库和逐步轨迹。
