# 50 条示范下 ALT 的训练轮数

## 研究问题

在 Lift-PH 用 50 条示范训练 ALT 时，延长训练能否提升闭环保持抬升的能力？10 条示范实验中，第 5 和第 10 轮在 24 局验证中均有 9 局保持达标，后期检查点只有 8 局验证。50 条实验用于检验早期持平或后期下滑的现象是否仍出现。

## 数据与方法

从同一冻结清单读取 50 条嵌套轨迹，包含此前 30 条及其前 10 条。20 条离线验证示范与训练池互斥。每个样本是两路 84×84 当前图像和 9 维当前机械臂状态。使用同一个 ImageNet 预训练 ResNet-18 处理两路图像，拼接本体状态特征，输出单位化的 128 维向量。用 NT-Xent 学习同帧轻微光度扰动间的一致性，训练期间更新整个编码器。Adam、批次 32、学习率 1e-4、温度 0.4 与 10 条实验一致。

在同一次训练中保存第 5、10、40、100 轮。每个检查点将这 50 条示范的全部观测重新编码为独立记忆库。推理每步使用当前观测查询余弦 top-1，执行匹配示范同一帧的 7 维增量动作。编码器不使用奖励、物体真值或未来帧。

## 评估

沿用 Lift-10 的 24 个验证初始状态，每局 250 步。指标为曾达抬升阈值、连续至少 20 步达阈值、第 250 步仍达阈值。连续保持是主指标。正式 50 个测试初始状态保留。按验证状态配对比较，并同时报告每个检查点的训练更新次数。同一轮数下，50 条示范带来更多更新，也扩大了记忆库覆盖，因此两种样本规模的差异需结合这些变化解读。本实验只训练 ALT，使用一个模型种子。

## 复现

在项目根目录用 Windows CUDA Python 和既有 WSL Lift 环境。复现时更换运行 ID。可在对应检查点写入后用 `--epoch` 提前验证单个轮数。

```powershell
$python = '..\alt_robomimic_lift\artifacts\environments\alt_lift_train\Scripts\python.exe'
& $python -B src/train_alt10.py alt50_epochs_v1 --scale 50
& $python -B src/evaluate_alt50.py alt50_epochs_v1 evaluate
& $python -B src/evaluate_alt50.py alt50_epochs_v1 summarize
```

| 研究环节 | 代码或配置 |
| --- | --- |
| 50 条嵌套清单校验和固定轮数 | `src/epoch_study.py`、[config.json](config.json) |
| 对比训练和逐检查点记忆库 | `src/train_alt10.py` |
| 复用 24 个初始状态的闭环验证 | `src/evaluate_alt50.py`、`src/evaluate.py` |
