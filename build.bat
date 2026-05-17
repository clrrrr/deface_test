@echo off
:: 双击此文件即可打包 transcode_gui.exe，无需手动打开终端

set CONDA=C:\Users\user\miniforge3\Scripts\conda.exe
set CONDA_ROOT=C:\Users\user\miniforge3

:: 激活 deface 环境
call "%CONDA_ROOT%\condabin\conda.bat" activate deface

pip install pyinstaller -q

for /f "delims=" %%i in ('python -c "import imageio_ffmpeg, os; print(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))"') do set FFMPEG_DIR=%%i

pyinstaller --onefile --windowed ^
  --add-binary "%FFMPEG_DIR%\*;imageio_ffmpeg\binaries" ^
  --collect-all imageio_ffmpeg ^
  transcode_gui.py

echo.
echo 打包完成，输出：dist\transcode_gui.exe
pause
