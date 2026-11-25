# 多视角点云配准 v9.6.4：CUDA 与可选手动架桥

本次以目标分支 `auto_register_multiview1_v9.1.2` 的提交 `9b1200aaea32c165dbb7d29d2d561c2b50054676` 为对照。该分支实际同时包含 v9.1.2、v9.3；两份历史脚本均保留。新增独立入口 `auto_register_multiview1_v9.6.4_gpu.py`，包含本地 v9.3.2 → v9.4 → v9.5 → v9.6.3 的累计改进，以及本次手动架桥恢复与保护。这不是只给远端 v9.3 增加一个开关。

## 与仓库现有版本对比

| 项目 | 远端 v9.1.2 | 远端 v9.3 | v9.6.4 |
|---|---|---|---|
| 计算 | CPU 顺序配准 | CPU 多进程窗口配对 | 共享预处理、CPU 初值与 CUDA 流水线、Float64 精确 KDTree / ICP |
| 初值 | RANSAC | RANSAC | 双向检验 FGR，失败回退 RANSAC；参考模式另有球几何初值 |
| RANSAC 迭代上限 | 200000 | 80000 | 80000；相对 v9.3 未减少 |
| 主图 | BFS、位姿图 / multiway ICP | `--source-main` 可选，默认关闭 | 默认 source-main + freeze-main；失败分量按刚体接入 |
| 数值与图优化 | 原实现 | 原实现 | 居中 SE(3) 步长、退化方向抑制、稳健图优化及几何回退 |
| 融合支持 | 邻域支持统计 | 邻域支持统计 | 精确排除自身帧，按独立视角计数 |
| 手动架桥 | 默认开启 | 默认开启 | 默认关闭；`--manual-bridge` 显式开启；兼容旧关闭参数 |
| 手动确认后 | 旧救援逻辑 | 旧救援逻辑 | 保留确认分量的内部相对位姿；跳过后续逐帧球 / 表面精修，支持撤销 |
| 参考引导 | 无专用球引导流程 | 无专用球引导流程 | 同一静态物体的实测四球参考；可加同坐标系表面参考 |
| 深度校正 | 无 | 无 | 仅 `--calibrate-depth` 显式开启，留出验证通过后使用 |
| 报告 | 基础评分 | 独立帧 / 边 / 闭环审计 | 继承审计，增加耗时、接入率、参考检查、深度模型、手动确认记录 |
| 运行限制 | 无 300 秒报告 | 无 300 秒报告 | 报告是否满足 300 秒；不按时间强制截断或丢帧 |

完整 `DEFAULT_CFG` 对照见 [参数表](docs/DEFAULT_CFG_COMPARISON.csv)，由三个实际源文件的 AST 提取，不执行历史脚本。表内“未定义”不等于关闭，旧实现可能在函数中使用回退值。CLI 参数另见 [CLI 对照](docs/CLI_COMPARISON.csv)。动态默认值保留源表达式；最终有效参数以本次输出 manifest 为准。

下列精度相关默认值相对 v9.3 保留：`hp_voxel=0.2`、`hp_icp_thr=[1.0,0.5,0.3]`、`hp_keep_ratio=0.7`、`hp_iter=150`、`score_voxel=0.5`、`merge_voxel=0.3`、`cover_radius=0.6`、`cover_min=3`。坐标默认按 mm 解释，输入采用其他单位时必须同步调整阈值。CUDA 搜索为精确搜索；接受候选的距离评分保持精确，已确定拒绝的评分可能被上界截断。Float64 并不表示与历史 CPU 优化路径逐位相同。

## 安装与运行

基础依赖仍使用仓库的 `pyproject.toml` / `uv.lock`，本次不改动锁文件：

```powershell
uv sync
uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt
.\run_v964_tests.cmd
```

GPU 附加依赖是 `cupy-cuda13x[ctk]==14.2.0`，需要兼容 CUDA 13 的 NVIDIA 驱动。`[ctk]` 安装所需 Toolkit 组件 wheel；同一环境不要混装多个 CuPy 包，详见 [CuPy 官方安装说明](https://docs.cupy.dev/en/stable/install.html)。后续直接调用启动脚本或 `.venv\Scripts\python.exe`；用 `uv run` 时加 `--no-sync`，避免重新同步时移除附加依赖。仅 CPU 运行可直接指定 `--backend cpu`，其耗时不能套用 GPU 基准。

默认使用仓库 `.venv`。有独立 GPU 环境时，将 `gpu_environment.example.cmd` 复制为 `gpu_environment.local.cmd`，按需指定 `REG_PYTHON`、`REG_GPU_DEPS`、`CUPY_CACHE_DIR`。私有配置已加入忽略列表，发布脚本没有硬编码本机环境路径。

### 不同物体或没有可靠参考

先只检查输入：

```powershell
.\run_gpu_no_reference.cmd "D:\data\object_b" "D:\results\object_b_v964" --check
```

全自动配准：

```powershell
.\run_gpu_no_reference.cmd "D:\data\object_b" "D:\results\object_b_v964"
```

保留手动架桥：

```powershell
.\run_gpu_no_reference.cmd "D:\data\object_b" "D:\results\object_b_v964_manual" --manual-bridge
```

直接调用 Python 等价入口：

```powershell
.\.venv\Scripts\python.exe .\auto_register_multiview1_v9.6.4_gpu.py "D:\data\object_b" --backend cuda --jobs 8 --output-dir "D:\results\object_b_v964_manual" --manual-bridge
```

原始帧放在输入文件夹顶层，脚本不递归搜子目录。已知的融合结果等生成 PLY 会被排除并打印，避免混入原始帧；其他自定义命名的参考或结果文件应放在输入文件夹之外。默认完整重算位姿；`--reuse-pairs` 只用于匹配当前输入和参数的帧对缓存。`--refine-existing` 是已有位姿精修，其耗时不能当作从原始帧开始的完整配准耗时。

### 同一物体的参考与可选校正

```powershell
.\run_gpu_calibrated.cmd "D:\data\object_a" "D:\refs\object_a_spheres.ply" "D:\refs\object_a_surface.ply" "D:\results\object_a_v964" --manual-bridge
```

该入口同时开启球引导、表面约束和深度校正。两份参考必须属于输入的同一静态物体，且已经处于同一参考坐标系；另一物体必须更换参考或改用无参考入口。参考文件应放在输入帧列表之外。先追加 `--check` 可只做预检，不打开选点窗口。

只需要刚体球引导、不开深度校正时：

```powershell
.\.venv\Scripts\python.exe .\auto_register_multiview1_v9.6.4_gpu.py "D:\data\object_a" --sphere-guide "D:\refs\object_a_spheres.ply" --output-dir "D:\results\object_a_rigid" --manual-bridge
```

`--surface-reference` 会参与优化，针对它的残差属于一致性检查。注册流程不读取计量 PDF。深度校正会改变输出点的几何，原始 PLY 不改写；重建时必须先执行 `depth_correct`，再应用 `transforms.npz` 中的刚体矩阵，不能直接把该矩阵乘到原始点上。请同时保留 `depth_calibration.json`、`coordinate_frame.json` 与 `transforms.npz`。本版本拒绝对校正结果直接执行 `--remerge` / `--metrics-only`，也不允许校正与 `--refine-existing` 混用，避免漏掉校正步骤。

## 手动架桥操作与保护范围

1. 自动配准、弱边和分量救援先执行；仍有失败分量时终端提示，输入 `y` 进入选点，直接回车跳过。
2. 在红色待接入分量和绿色目标主云中，按相同顺序各选至少 4 个对应点；建议 5–6 个，分布尽量分散，不能共线。各自完成后关闭窗口。
3. 查看叠加效果，终端 `y` 确认、`r` 重选、`n` 放弃。在后续救援提示处 `u` 撤销上一次手动接入。
4. 组内多帧作为一个刚体接入。手动目标采用初始主云快照；确认后保留分量内部相对位姿，保护其确认姿态不被可选救援后全局 polish 和后续球 / 表面逐帧精修覆盖。输出坐标系的整体刚体换基仍会一致应用到所有帧。

`--manual-bridge` 与 `--no-rescue` / `--no-manual-rescue` 互斥。关闭参数仅关闭人工选点，自动救援仍执行。显式手动开关在通用 `--set` 之后生效。启用手动选项需要 PyVista / VTK；自动全部接入成功时不会要求选点。

新增 `manual_bridge_audit.json` 记录开关、确认帧索引和最终导出坐标系下的矩阵。`performance.json` 增加 `manual_bridge_enabled`、`manual_bridge_frames`、`timing_mode`。人工等待计入总耗时，因此交互模式不承诺五分钟完成。选择了错误对应点仍可能产生错误位姿，确认前应检查叠加效果。

## 验证与证据边界

本次 v9.6.4 验证：40 项测试通过，包括 7 项球几何、7 项深度校正、12 项 CUDA 数值、10 项手动架桥、4 项 Windows 启动脚本测试。CUDA 检查使用精确 CPU 参考、退化点云、边界距离、并发查询、自帧排除及 ICP；手动测试检查组内刚性、主云冻结、确认后保护、撤销和导出矩阵。启动测试覆盖带空格路径及参数转发。另用实际 50 个原始 PLY 完成无参考入口的预检，排除 5 个生成 PLY，坏帧 0。

实际 PyVista 点云转换已测试；本次未进行人工窗口点选，也未重新跑 v9.6.4 的整批配准。因此下面是历史 v9.6.3 算法结果，不能宣称为 v9.6.4 新实测，或与远端 v9.3 的同条件耗时对比。

| 历史本地运行 | 输入 / 接入 | 完整耗时 | 条件 |
|---|---:|---:|---|
| v9.5.0 | 213 / 187 | 728.235 s | CUDA 无参考全流程，未请求帧对复用 |
| v9.6.3 | 213 / 213 | 294.188 s | CUDA 四球 + 表面参考 + 深度校正；新的预处理 / 帧对 / 位姿 / CuPy 缓存 |

213 个文件含 6 份完全重复数据，独立内容为 207 帧。历史机器为 RTX 5060 Laptop GPU；不同机器、数据量和模式不保证同一耗时。300 秒是报告目标，不是裁剪任务的时间上限。

相同 187 帧、相同四球拟合阈值 0.5 mm 的历史计量对比：球心距离 RMSE 从 v9.5.0 的 0.211029 mm 降至 v9.6.3 的 0.094252 mm，直径 MAE 从 0.422162 mm 降至 0.230063 mm。v9.6.3 全 213 帧分别为 0.098000 mm 和 0.234021 mm。计量 PDF 未用于优化，但这只验证球几何；部分额外整体对齐后的表面指标略有回退，最佳单帧的球几何也仍优于融合结果，不能推断全场景、所有指标或其他物体精度均已提升。

可公开的精简验证摘要见 [validation_v964.json](docs/validation_v964.json)。不随代码上传原始数据、参考模型、计量 PDF、位姿缓存或点云输出。

## 本次提交范围

- 新增 v9.6.4 主脚本、可移植启动器、GPU 附加依赖、五组测试及本说明；README 增加最新入口。
- 保留远端 v9.1.2 / v9.3、原单帧脚本、`pyproject.toml`、`uv.lock` 和历史说明，便于复现。
- 增加缓存、运行输出和本机私有配置的 Git 忽略规则；不移除仓库已经跟踪的数据文件。
- PR 的目标是用户指定的 `auto_register_multiview1_v9.1.2`，提交放在独立功能分支中供审阅。
