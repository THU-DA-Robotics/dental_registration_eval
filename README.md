\# dental\_registration\_eval



牙齿点云配准与车针位姿自动求解。



\## 环境配置



本项目使用 \[uv](https://docs.astral.sh/uv/) 管理 Python 环境与依赖。



1\. 安装 uv（见 https://docs.astral.sh/uv/getting-started/installation/ ）

2\. 克隆仓库后，一键创建环境并安装依赖：



&#x20;  uv sync



\## 运行



&#x20;  uv run python auto\_register.py



\## 数据说明



运行需要原始口扫（mesh）与含车针的点云文件，放在项目根目录，

并在 auto\_register.py 开头的 SCAN\_PATH / CLOUD\_PATH 中配置文件名。

数据文件体积较大，不纳入版本管理。




## 最新多视角版本：v9.6.4

新增 CUDA Float64 多视角配准、同物体参考引导及可选手动架桥；保留 v9.1.2 和 v9.3 历史入口。

- [升级说明、版本及参数对比、运行示例与验证边界](V9_6_4_UPGRADE_NOTES.md)
- 最新脚本：`auto_register_multiview1_v9.6.4_gpu.py`
- 无参考：`run_gpu_no_reference.cmd "输入目录" "输出目录"`
- 保留人工选点：在命令末尾加 `--manual-bridge`；只检查输入加 `--check`。
- 参考校正：`run_gpu_calibrated.cmd "输入目录" "对应四球参考.ply" "同坐标系表面参考.ply" "输出目录"`。另一物体请更换参考或使用无参考入口。
- 基础依赖：`uv sync`；GPU 附加依赖：`uv pip install --python .venv\Scripts\python.exe -r requirements-gpu.txt`。
- 回归检查：`run_v964_tests.cmd`；不运行 CUDA 数值测试时用 `run_v964_tests.cmd --cpu-only`。

手动等待计入耗时。历史 294.188 秒数据属于 v9.6.3 指定参考模式，不代表所有数据或 v9.6.4 已实测达到五分钟。
