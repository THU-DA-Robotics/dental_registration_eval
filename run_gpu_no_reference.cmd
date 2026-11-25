@echo off
setlocal
chcp 65001 >nul
if "%~1"=="" goto help
if /I "%~1"=="--help" goto help
call "%~dp0_gpu_environment.cmd"
if errorlevel 1 exit /b 1
set "REG_SCRIPT=%~dp0auto_register_multiview1_v9.6.4_gpu.py"
set "REG_DATA=%~f1"
set "REG_OUTPUT=%REG_DATA%\multiview_output_v964_no_reference"
if not "%~2"=="" set "REG_OUTPUT=%~f2"
set "REG_MANUAL=--no-manual-rescue"
set "REG_CHECK="
:options
if "%~3"=="" goto ready
if /I "%~3"=="--manual-bridge" (set "REG_MANUAL=--manual-bridge"& shift /3& goto options)
if /I "%~3"=="--check" (set "REG_CHECK=--precheck-only"& shift /3& goto options)
echo 可选参数只支持 --manual-bridge 或 --check。
exit /b 1
:ready
if not exist "%REG_DATA%\" (echo 输入目录不存在，请检查第一个参数。& exit /b 1)
if exist "%REG_OUTPUT%\depth_calibration.json" (echo 输出目录含深度校正结果，请指定新目录。& exit /b 1)
if exist "%REG_OUTPUT%\sphere_guide.json" (echo 输出目录含参考引导结果，请指定新目录。& exit /b 1)
echo [模式] 无参考GPU刚体配准；默认全自动。
if "%REG_MANUAL%"=="--manual-bridge" echo [手动架桥] 开启；自动接入失败后询问是否选点。
"%REG_PYTHON%" "%REG_SCRIPT%" "%REG_DATA%" --backend cuda --output-dir "%REG_OUTPUT%" --jobs 8 %REG_MANUAL% %REG_CHECK%
exit /b %ERRORLEVEL%
:help
echo 用法：run_gpu_no_reference.cmd "输入目录" "输出目录" [--manual-bridge] [--check]
echo --manual-bridge：自动接入失败后允许手动点选架桥。
echo --check：只检查输入，不执行配准或打开选点窗口。
exit /b 0
