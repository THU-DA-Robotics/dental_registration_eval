@echo off
rem Called from a launcher with SETLOCAL. Private overrides are never committed.
set "REG_ROOT=%~dp0"
if exist "%REG_ROOT%gpu_environment.local.cmd" call "%REG_ROOT%gpu_environment.local.cmd"
if not defined REG_PYTHON set "REG_PYTHON=%REG_ROOT%.venv\Scripts\python.exe"
if not exist "%REG_PYTHON%" (echo Project Python missing. Run uv sync or set REG_PYTHON.& exit /b 1)
if defined REG_GPU_DEPS if not exist "%REG_GPU_DEPS%\cupy" (echo REG_GPU_DEPS does not contain CuPy.& exit /b 1)
if defined REG_GPU_DEPS set "PYTHONPATH=%REG_GPU_DEPS%;%PYTHONPATH%"
if not defined CUPY_CACHE_DIR set "CUPY_CACHE_DIR=%REG_ROOT%.cupy_cache"
set "PYTHONIOENCODING=utf-8"
exit /b 0
