# PyInstaller spec for transcode_gui (Windows one-folder build).
# Build with: pyinstaller transcode_gui.spec --clean --noconfirm
#
# Notes:
# - One-folder mode avoids extraction to temp dir and file-not-found issues
# - vendor/ffprobe.exe must be present before building

from PyInstaller.utils.hooks import collect_all
from pathlib import Path
import os

block_cipher = None
PROJECT_ROOT = Path(SPECPATH).resolve()

# Collect imageio_ffmpeg
imageio_datas, imageio_binaries, imageio_hidden = collect_all('imageio_ffmpeg')

# Add ffprobe.exe from vendor
added_datas = list(imageio_datas) + [
    (str(PROJECT_ROOT / 'vendor' / 'ffprobe.exe'), 'vendor'),
]

a = Analysis(
    ['transcode_gui.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=imageio_binaries,
    datas=added_datas,
    hiddenimports=imageio_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='transcode_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='transcode_gui',
)
