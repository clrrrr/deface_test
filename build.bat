@echo off
:: 双击此文件即可打包 transcode_gui.exe，无需手动打开终端

set CONDA=C:\Users\user\miniforge3\Scripts\conda.exe
set CONDA_ROOT=C:\Users\user\miniforge3

:: 检查 vendor\ffprobe.exe 是否存在
if not exist "vendor\ffprobe.exe" (
    echo [ERROR] vendor\ffprobe.exe 不存在
    echo 请先将 ffprobe.exe 放入 vendor\ 目录
    pause
    exit /b 1
)

:: 激活 deface 环境
call "%CONDA_ROOT%\condabin\conda.bat" activate deface

pip install pyinstaller -q

:: 清理之前的构建
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

:: 使用 spec 文件打包
pyinstaller transcode_gui.spec --clean --noconfirm

echo.
echo 打包完成，输出：dist\transcode_gui\
echo 运行：dist\transcode_gui\transcode_gui.exe
pause
