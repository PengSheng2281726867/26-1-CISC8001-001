# Lift 旧版仿真环境配置记录

## 目标

按用户要求，安装与已下载 Lift/PH/low_dim 数据一致的仿真环境，并验证真实示范回放。
仿真依赖、引擎、源码包、环境、缓存和大型产物都保存到 Git 忽略的 artifacts/environments/robomimic_lift_legacy/。
没有网络训练或 DP 训练，没有更改数据文件或旧 Push-T 实验。

## 已核实依据

- 数据 SHA256：18106c8c072020672fecdd2a9ffbfe7dc656823cb877f972ade7fad34d2a5b74。
- 使用本地 robomimic 0.2.0 下载的旧版数据：Lift、Panda、OSC_POSE、control_delta=True、20Hz。
- 文件共 200 Episode、9,666 帧；原始 mask/train=180、mask/valid=20，不相交且覆盖全部 Episode；保留现有划分。
- 旧版官方数据文档要求 robosuite offline_study 分支；其当前冻结 Commit 为 d9baa17d5328d7b0ec15ecda54c76c61bb983124，声明版本 1.2.0。
- upstream setup.py 固定 mujoco-py==2.0.2.9，而同分支 requirements.txt 写 2.0.2.13；初始选择遵循包声明 2.0.2.9，记录这一上游不一致，实际回放将验证兼容性。
- MuJoCo 2.0 由官方提供公共免费激活密钥（2031-10-18 到期），仅本地保存，不提交密钥。
- Windows 现有 Push-T Python 环境不具备 robosuite/MuJoCo；旧版官方安装支持 Linux/macOS，因此使用现有 WSL2 Ubuntu 26.04。
- WSL 中原有 Python 3.14.4 不适合此旧依赖链；在项目内另建 Python 3.8.20 环境，保留原环境。
- 经工具批准安装 WSL 系统编译和 Mesa 渲染依赖：build-essential、libosmesa6-dev、libgl1-mesa-dev、libglew-dev、patchelf。APT 日志可见会话工具输出。

## 验证方案

固定 demo_0、demo_1、demo_2：逐帧恢复状态，比较文件中的全部低维观测；另从初始状态连续回放动作，独立记录动态状态误差和任务成功。
验证 agentview 与 robot0_eye_in_hand 相机的离屏渲染，保存首条示范的初/中/末图。
不能以 import 成功代替回放通过；失败、修复和实际结果继续记录在下方。

## 安装启动诊断

1. 直接运行 run.ps1 被本机 PowerShell 的脚本执行策略拒绝，尚未执行安装。改为使用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...`，只改变当前进程的策略，没有改动系统/用户执行策略。
2. 启动器最初只显示路径转换失败，初步怀疑 Windows 反斜杠，统一为正斜杠后仍失败。加入退出码/参数诊断后确认根因是受限子 PowerShell 中 WSL 服务返回 E_ACCESSDENIED，直接 WSL 命令可正常工作。未修改系统权限或账户，安装使用获准的 WSL 调用；启动器稍后在授权环境中验证。
3. 下载 uv 压缩包至约 4 MB 时遇到 curl SSL unexpected EOF。保留部分下载，修改下载函数为 .partial 文件加断点续传、最多三次传输重试，成功后才重命名为最终文件，避免把残缺压缩包误认为已完成。
   在授权环境下续传成功，uv 已正常解压并安装 Python 3.8.20。uv 默认在用户 .local/bin 建立 python3.8 软链接；后续设置 UV_PYTHON_BIN_DIR 到项目内 bin，准备只移除此次创建、目标已核实指向本项目的外部软链接。
   已经核实软链接目标，并通过授权操作将其移入项目 bin；后续安装不会再生成外部 Python 链接。
4. mujoco-py 2.0.2.9 的 PEP517 隔离构建自行安装 Cython 3，忽略运行环境中要求的 0.29.36，导致回调函数类型的 Cython 编译错误。修复：先安装锁定的构建依赖，再用 --no-build-isolation 构建原版 mujoco-py 2.0.2.9。未修改 MuJoCo/robosuite 源码或物理参数；原失败日志保存在 setup.log。
5. 修复 Cython 后，GCC 15 将旧版生成 C 代码中匿名结构体指针的类型警告视为错误，导致绑定编译失败。只为此诊断加入 `-Wno-error=incompatible-pointer-types`，恢复旧编译器对此项警告的处理方式；不修改生成代码，不加入 fast-math，不更改预编译 MuJoCo 物理库。验证结果会记录引擎与绑定的 SHA256。
6. 上一步已使核心绑定 C 文件通过编译；随后旧版 osmesashim.c 缺少 stdio.h，调用 printf 在新 GCC 下成为错误。用编译参数 `-include stdio.h` 补充标准头声明，保持原始包源码和仿真算法不变，再构建绑定。
7. 核心和渲染代码编译已通过，链接阶段因 Python standalone 的编译器名为 cc，旧 distutils 未识别其为 GCC，生成了 GCC 不支持的 -R 选项。显式设置 CC=gcc、CXX=g++，使旧 distutils 使用正确的 GNU 链接参数；再次保留原版本重建。
   实际检查显示，仅设置 CC 未能解决：旧 distutils 仍根据 Python 的编译时 CC 配置生成 -R，且链接器沿用 cc。进一步把 LDSHARED 指向项目内的窄范围链接适配器，只将 -R/path 转为 GNU 的 -Wl,-rpath,/path，其余参数原样转发至 gcc；不修改上游包或 Python 的全局 sysconfig。
   最小共享库编译/链接/加载检查返回预期值 7 后，完整 mujoco-py 构建成功，安装保留为 2.0.2.9。随后 robosuite 1.2.0 / robomimic 0.2.0 / CPU Torch 1.10.2 / torchvision 0.11.3 安装成功；2026-10-22T05:42:48Z 导入检查通过。
8. 审查原版环境接口发现：离屏模式下会导入 egl_probe，即使使用 CPU。补齐 EGL 探测和它的 CMake 构建依赖，以及 robomimic 声明的 TensorBoard 日志依赖；仅安装工具，不启动训练。独立执行 extras 阶段后运行 pip check，再开始真实回放。
   CMake wheel 下载中途截断（26.1 MB 仅收到约 2.2 MB），pip 的 SHA256 校验拒绝安装。使用可断点续传的下载命令获取同一个官方 wheel，显式核对 PyPI 提供的 SHA256 后才允许从本地安装；未关闭 SSL 或哈希验证。
   补齐后 pip check 返回 No broken requirements found，所有安装声明依赖完整。
9. 首次通过授权的 PowerShell 启动器运行验证，WSL 路径转换正常，确认先前是沙箱子进程访问限制。验证在导入 Torch 时因旧 libtorch_cpu.so 的可执行栈标记被新 glibc 拒绝，尚未创建环境或回放。仅在新建环境内备份原库并用 patchelf --clear-execstack 清除该标记，记录前后哈希和差异字节；不触及 MuJoCo 或旧 Push-T 环境。修复后先检查 Torch 导入和简单 CPU 张量，再重新做回放验证。
   修复实测只改变一个字节，库文件大小不变；Torch 1.10.2+cpu 导入与张量计算通过，原件及前后 SHA256 保存在本地。后续验证把修复记录一起保存。
10. 2026-10-22T06:02:08Z 验证运行：三条示范均连续回放成功，恢复状态后的观测最大误差 1.67e-16，两台相机出图正常；但 demo_1 / demo_2 的连续完整状态最大误差为 8.75e-4 / 2.65e-4，超过预先设置的严格 1e-5 门限，命令返回失败。原结果另存 results/initial_validation.json，没有放宽门限。
    为区分动力学差异、速度分量与观测缓存时序，下一次运行增加 qpos/qvel、物体位姿分量、误差峰值索引及逐步轨迹记录。另用独立 MjSim 从当前状态计算几何观测，避免为诊断向实际回放额外插入 forward() 或修改仿真逻辑；原始逐步返回观测误差仍保留。
11. 2026-10-22T06:04:46Z 复核回放：三条示范再次成功，完整状态最大误差与首次逐项一致，说明新增只读诊断没有改变回放。demo_1 / demo_2 的峰值均为动作索引 48、展平状态索引 29，即物体角速度第一个分量；物体位置最大欧氏差分别为 0.003139 mm、0.000553 mm，姿态差分别为 0.005768°、0.000926°。状态恢复、动作执行与渲染可用，但未达到严格完整状态门限，原门限和非零退出状态保留。
    独立状态对齐后，object 观测分量最大误差从 step 返回值的约 0.0013 降至 3.60e-10 / 5.02e-5 / 7.37e-6，说明观测更新时间是另一项需要区分的因素；没有修改原环境的观测接口。误差在接触后放大的根因仍可能涉及数值精度或隐藏内部量，不能宣称已证明唯一原因。
12. 使用独立 summarize_replay.py 读取保存的实际状态数组，核对首次超限索引、位置欧氏误差、四元数周期角度误差、峰值分量和任务标志。此步骤不运行仿真、不读取新的测试状态，不改变训练/验证划分。两台相机拼图已人工检查，物体、夹爪及抬升画面正常。

## 交付与 Git

配置完成范围为本机 WSL 独立旧版环境、可重跑安装/验证程序和真实回放诊断；不把严格回放检查的未通过记录为全面通过。
数据文件完整 SHA256 未变；ACl/alt 和 DP/diffusion_policy 在只读检查中均没有工作树变更。检查临时指定 safe.directory，未修改其 Git 配置或系统 ACL。
仅将 environments/robomimic_lift 下脚本、配置、中文说明、小型统计与拼图纳入此次提交；环境、数据、引擎、激活文件和完整状态轨迹保持 Git 忽略。
提交和推送状态以 Git 历史及会话实际返回值为准；本文件不写入自指提交哈希。
