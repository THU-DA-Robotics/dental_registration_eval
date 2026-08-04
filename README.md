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

