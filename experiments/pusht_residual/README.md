# Push-T 检索动作的学习修正

研究问题是从训练示范检索到一段动作后，小网络能否修正几何偏差，并改善闭环推物。两组实验都使用 20 条预留示范和固定的 50 个正式测试种子。几何检索及动作变换使用仿真真值。

| 实验 | 状态 | 入口 |
| --- | --- | --- |
| 检索加残差的三轮闭环对照 | 已完成并复核 450 局 | [三轮综合报告](experiments/residual_study/REPORT.md) · [方法与命令](experiments/residual_study/EXPERIMENT.md) · [结果](experiments/residual_study/RESULTS.md) · [实验记录](experiments/residual_study/LOG.md) |
| 残差网络的训练轮数、监督规模与输出结构 | 已完成三轮并复核 296 局 | [三轮综合结果](experiments/network_study/RESULTS.md) · [实验方法](experiments/network_study/EXPERIMENT.md) · [实验记录](experiments/network_study/LOG.md) |

前一组实验依次测试全局检索、推杆相对 T 的检索、以及限制残差幅度。相对几何检索加坐标对齐得 0.9261，有界残差得 0.6179，无界残差得 0.3222。新一组固定 30 条检索库，比较网络训练设置。正式测试中衰减残差得 0.8617，纯对齐得 0.9261。

本目录归档了原 `push_T_alt` 的两组实验。代码复用本仓库的数据适配器与冻结清单，以及同级 `DP` 项目的原版模拟器。原始运行的代码和数据校验和保留在证据中。归档版只调整了仓库定位和历史对齐结果的读取路径。权重和完整查询轨迹留在原实验目录，复现会在本目录生成 Git 忽略的 `artifacts/`。

视觉 ALT 与 DP 的匹配对照见 [Push-T 视觉比较](../pusht_dp/README.md)。
