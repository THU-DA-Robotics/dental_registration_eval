@echo off
setlocal
call "%~dp0_gpu_environment.cmd"
if errorlevel 1 exit /b 1
if /I "%~1"=="--cpu-only" goto cpu
"%REG_PYTHON%" -m unittest discover -s "%~dp0." -p "test_*_v964.py" -v
exit /b %ERRORLEVEL%
:cpu
pushd "%~dp0."
"%REG_PYTHON%" -m unittest test_geometry_v964 test_depth_v964 test_manual_bridge_v964 test_launchers_v964 -v
set "REG_STATUS=%ERRORLEVEL%"
popd
exit /b %REG_STATUS%
