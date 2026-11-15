# 汇报短片

两段短片来自已完成实验的本地仿真录像，作为 [GitHub Release 附件](https://github.com/PengSheng2281726867/26-1-CISC8001-001/releases/tag/presentation-media-v1) 发布。PPT 可先下载 MP4，再插入对应页。时间跳转保留原控制步数，案例用于解释指标。原始完整录像与动作日志留在本地。

<a id="lift-alt30"></a>

## 第 3 页｜Lift

**ALT‑30，种子 100049，11.1 秒。** 素材来自原始独立 ALT‑30 的 `evaluation01`，第 40 步首次抬过高度阈值，第 1,000 步低于阈值。保留第 0–150 步与第 930–1,000 步。正文表格的正式对比使用后续统一评估器的数据。

[下载 Lift 短片](https://github.com/PengSheng2281726867/26-1-CISC8001-001/releases/download/presentation-media-v1/lift-alt30-seed100049.mp4) · [PPT 封面帧](materials/lift/lift_video_still.png) · [该录像的逐局记录](../../experiments/alt_robomimic_lift/experiments/runs/alt_lift_ph30_image_proprio_seed42_v1/evaluations/evaluation01/eval_rollouts.csv)

<a id="push-t-aligned"></a>

## 第 7 页｜Push‑T

**相对几何检索＋坐标对齐，种子 100003，13.1 秒。** 使用仿真真值 T 位姿。第 201 步的最高覆盖得分是 0.930，末步得分是 0.203。保留第 159–239 步与第 251–300 步。以原动作日志复现全部 300 步，检查峰值和末步分，再用渲染帧确认原始录像的种子。

[下载 Push‑T 短片](https://github.com/PengSheng2281726867/26-1-CISC8001-001/releases/download/presentation-media-v1/pusht-aligned-seed100003.mp4) · [PPT 封面帧](materials/pusht_geometry/pusht_video_still.png) · [该录像的逐局记录](../../experiments/pusht_residual/experiments/residual_study/evidence/round3_evaluate_aligned.csv)

## 素材来源与核验

[媒体清单](materials/media_manifest.json)保存完整录像和短片的 SHA‑256、剪辑区间及实验记录路径。已有短片可从 Release 直接下载。重剪需保留原始 ALT‑30 录像和同级 `push_T_alt` 的第 3 轮对齐录像与查询日志。从仓库根目录执行：

```powershell
$python = '..\DP\.conda\pusht\python.exe'
& $python -B docs/presentation/build_clips.py --output artifacts/presentation/clips_reproduction
```

可用 `--residual-aligned` 指定本地原实验目录。重剪会生成短片、封面帧和带校验和的清单。不同编码器版本生成的 MP4 字节可能有差异。
