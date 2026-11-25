@echo off
rem Optional private overrides. Copy to gpu_environment.local.cmd only if needed.
rem Normal installations use .venv\Scripts\python.exe and its installed CuPy.
rem set "REG_PYTHON=C:\path\to\python.exe"
rem set "REG_GPU_DEPS=C:\path\to\external_gpu_packages"
rem set "CUPY_CACHE_DIR=C:\path\to\cupy_cache"
rem Default calibrated data is optional and must belong to ONE static object.
rem set "REG_DEFAULT_DATA=D:\data\object_a"
rem set "REG_DEFAULT_SPHERE_GUIDE=D:\references\object_a_spheres.ply"
rem set "REG_DEFAULT_SURFACE_REFERENCE=D:\references\object_a_surface.ply"
