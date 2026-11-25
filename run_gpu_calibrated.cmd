@echo off
setlocal
chcp 65001 >nul
if /I "%~1"=="--help" goto help
call "%~dp0_gpu_environment.cmd"
if errorlevel 1 exit /b 1
set "REG_MANUAL=--no-manual-rescue"
set "REG_CHECK="
if /I "%~1"=="--manual-bridge" (set "REG_MANUAL=--manual-bridge"& shift)
if "%~1"=="" goto defaults
if "%~3"=="" (echo 自定义数据需要输入目录、对应四球参考及口扫参考。& exit /b 1)
set "REG_DATA=%~f1"
set "REG_GUIDE=%~f2"
set "REG_SURFACE=%~f3"
set "REG_OUTPUT=%REG_DATA%\multiview_output_v964_calibrated"
if not "%~4"=="" set "REG_OUTPUT=%~f4"
goto options
:defaults
if not defined REG_DEFAULT_DATA goto help
if not defined REG_DEFAULT_SPHERE_GUIDE goto help
if not defined REG_DEFAULT_SURFACE_REFERENCE goto help
set "REG_DATA=%REG_DEFAULT_DATA%"
set "REG_GUIDE=%REG_DEFAULT_SPHERE_GUIDE%"
set "REG_SURFACE=%REG_DEFAULT_SURFACE_REFERENCE%"
set "REG_OUTPUT=%REG_DATA%\multiview_output_v964_calibrated"
:options
if "%~5"=="" goto ready
if /I "%~5"=="--manual-bridge" (set "REG_MANUAL=--manual-bridge"& shift /5& goto options)
if /I "%~5"=="--check" (set "REG_CHECK=--precheck-only"& shift /5& goto options)
echo 可选参数只支持 --manual-bridge 或 --check。
exit /b 1
:ready
if not exist "%REG_DATA%\" (echo 输入目录不存在。& exit /b 1)
if not exist "%REG_GUIDE%" (echo 四球参考PLY不存在。& exit /b 1)
if not exist "%REG_SURFACE%" (echo 口扫参考PLY不存在。& exit /b 1)
if "%REG_MANUAL%"=="--manual-bridge" echo [手动架桥] 开启；人工等待计入总耗时。
"%REG_PYTHON%" "%~dp0auto_register_multiview1_v9.6.4_gpu.py" "%REG_DATA%" --sphere-guide "%REG_GUIDE%" --calibrate-depth --surface-reference "%REG_SURFACE%" --jobs 8 --output-dir "%REG_OUTPUT%" %REG_MANUAL% %REG_CHECK%
exit /b %ERRORLEVEL%
:help
echo 用法：run_gpu_calibrated.cmd "输入目录" "四球参考.ply" "口扫参考.ply" "输出目录" [--manual-bridge] [--check]
echo 两份参考必须属于同一静态物体且在同一坐标系；另一物体请用无参考入口。
echo 可在私有gpu_environment.local.cmd中配置默认数据；不带参数时才使用它们。
exit /b 0
