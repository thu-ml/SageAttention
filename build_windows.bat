@echo off
setlocal enabledelayedexpansion
rem Build a SageAttention wheel on Windows with MSVC and CUDA 12.x/13.x.
rem Run from an "x64 Native Tools Command Prompt for VS 2022", or let this
rem script find vcvars64.bat in the default VS Community location.
rem
rem Overridable env vars:
rem   TORCH_CUDA_ARCH_LIST  target GPU archs (default 8.9 = Ada / RTX 40xx)
rem   EXT_PARALLEL          extensions built in parallel (default 3)
rem   MAX_JOBS              ninja jobs per extension (default 4)

where cl >nul 2>nul
if errorlevel 1 (
    set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if not exist "!VCVARS!" (
        echo error: cl.exe is not on PATH and vcvars64.bat was not found.
        echo Run this from an "x64 Native Tools Command Prompt for VS 2022".
        exit /b 1
    )
    call "!VCVARS!" >nul
)

set DISTUTILS_USE_SDK=1
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=8.9
if "%EXT_PARALLEL%"=="" set EXT_PARALLEL=3
if "%MAX_JOBS%"=="" set MAX_JOBS=4

rem Tag the wheel with the torch/CUDA pair it is pinned to, e.g. 2.2.0+cu132torch2.13.0
for /f %%v in ('python -c "import torch; print(torch.__version__.split('+')[0])"') do set TORCH_VER=%%v
for /f %%v in ('python -c "import torch; print(torch.version.cuda.replace('.', ''))"') do set CU_VER=%%v
set SAGEATTENTION_WHEEL_VERSION=2.2.0+cu%CU_VER%torch%TORCH_VER%

cd /d %~dp0
if exist build rmdir /s /q build
python setup.py bdist_wheel
if errorlevel 1 exit /b 1
echo.
echo Wheel written to %~dp0dist
