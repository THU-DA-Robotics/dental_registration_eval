# 多视角点云配准 v9.3 — 升级说明

> 本版本是 `auto_register_multiview1_v9.1.2.py` 的后续迭代。保留旧版脚本用于重现和 A/B 对照，**不直接覆盖 v9.1.2**。
>
> 新版文件：`auto_register_multiview1_v9.3.py`（来自已获较好效果的 `registration_source_main_v2_metrics.py`，仅重命名，未改其算法正文）。

## 一、与 v9.1.2 的主要区别

| 项目 | v9.1.2 | v9.3 |
|---|---|---|
| 两两注册 | 单进程顺序 RANSAC / ICP | 支持多进程窗口式配对、缓存、worker 回收及超时重试 |
| RANSAC 默认迭代 | 200,000 | 80,000（算法参数有变化，不能认为与原版数值结果相同） |
| 主点云 | BFS 初值 + Pose Graph + 全体 multiway ICP | 增加 `--source-main`：只选一个连通分量，分量内部 BFS 位姿，直接构建主云，跳过跨分量自动架桥和初始全局优化 |
| 主图选择 | 全局选锚点 | 增加 `--seed-frame`、`--seed-component` 显式选择 |
| 自动救援 | 弱边级联和分量救援 | 继承并改进；增加可选 `--freeze-main`，阻止救援后再次改变初始主分量位姿 |
| 局部守门 | 按旧逻辑执行 | 默认诊断模式；另有 `--gate-enforce`、`--gate-legacy`、`--gate-off` |
| 文件稳定性 | 基础读取 | PLY 预检、异常输出、进程重试 |
| 重复运行 | 重跑计算 | `--reuse-pairs` 复用匹配/参数一致的两两配准缓存 |
| 误差评价 | 主要是中位残差和重叠率 | `--metrics-only`、逐帧交叉残差、双向边残差、三角闭环报告 |

> 特别注意：v9.3 来源于 v9.1.2 → v9.1.5 → Source-Main → Metrics 的多轮迭代，并不是在 v9.1.2 上只修改几行。实际配准质量需要按相同数据集对照；“重叠区最近邻残差”不是有真值标定的绝对精度。

## 二、推荐用法（Windows / uv）

在仓库根目录执行：

```powershell
uv run python auto_register_multiview1_v9.3.py "D:\explore\scientifical_research\MeshData_1023" --jobs=8 --source-main --snapshot-initial
```

第二次可加 `--reuse-pairs` 避免重复计算 RANSAC：

```powershell
uv run python auto_register_multiview1_v9.3.py "D:\explore\scientifical_research\MeshData_1023" --jobs=8 --source-main --reuse-pairs --snapshot-initial
```

已有完整结果可直接只生成误差评价（无需重新配准）：

```powershell
uv run python auto_register_multiview1_v9.3.py "D:\explore\scientifical_research\MeshData_1023" --metrics-only
```

选择已知对齐较好的源分量：

```powershell
uv run python auto_register_multiview1_v9.3.py "D:\explore\scientifical_research\MeshData_1023" --jobs=8 --source-main --seed-component 1 --reuse-pairs
```

主图冻结（可选；会改变救援后的优化流程）：

```powershell
uv run python auto_register_multiview1_v9.3.py "D:\explore\scientifical_research\MeshData_1023" --jobs=8 --source-main --freeze-main --reuse-pairs
```

## 三、定量评价输出

默认位于输入文件夹下的 `multiview_output/`：

- `registration_metrics_frames.csv`：逐帧跨帧重叠率、未匹配率、匹配区域 Median / RMSE / P90 / P95。
- `registration_metrics_edges.csv`：配对双向残差以及位姿图残差。
- `registration_metrics_cycles.csv`：由原始配准边构成的三角闭环一致性。
- `registration_metrics_summary.json`：总体指标和单位、抽样、绝对精度警告。
- `registration_run_manifest.json`：文件列表与参数，帮助确认旧位姿是否仍匹配当前数据。

**误差注意事项**：计量单位取决于输入 PLY 坐标，默认假设 mm；误差是在抽样、最近邻、阈值内重叠点上测量的统计一致性，不构成对真值的验证。请同时检查覆盖率与未匹配率。

## 四、兼容和回归验证

1. 已检查 Python 语法；未在本执行环境中安装 Open3D，尚未对真实 211 帧数据完成运行验证。
2. 本次提交采用**新增文件**方式，不替换仓库的 `auto_register_multiview1_v9.1.2.py`、`auto_register.py`、`pyproject.toml`、`uv.lock`。
3. 启用 `--source-main` 才切换为经过实际使用反馈表现较好的源分量主图策略；`--freeze-main`、`--gate-enforce` 均需显式开启。
4. 如果原实验曾指定 `--seed-frame` 或 `--seed-component`，新版运行时继续使用同一个参数才能公平比较。

## 五、提交到现有分支

ZIP 还提供 `push_v9_3.ps1`：在仓库根目录解压后运行，脚本会核对分支和远端、检查 Python 语法、只提交新版脚本和本升级说明，并执行推送。需在本机事先完成 GitHub 登录并具备仓库写入权限。

```powershell
git clone --branch auto_register_multiview1_v9.1.2 https://github.com/THU-DA-Robotics/dental_registration_eval.git
cd dental_registration_eval
# 将 ZIP 解压到当前目录，然后执行
.\push_v9_3.ps1
```

## 六、推荐提交说明

```text
feat(registration): add v9.3 source-main alignment and quantitative metrics

- preserve v9.1.2 as a reproducible baseline
- add source-component BFS main-cloud construction
- improve pairwise worker reliability and cached reruns
- add per-frame / bidirectional / cycle consistency reports
- make freeze-main and strict gate opt-in
```
