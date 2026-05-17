# 视频转码工具

GPU 加速视频转码工具，基于 FFmpeg。提供命令行（transcode.py）和 GUI（transcode_gui.exe）两种使用方式。

---

## GUI 使用（Windows，推荐）

### 直接使用
双击 `dist/transcode_gui.exe`，无需安装 Python 或 conda。

界面说明：
- **输入文件夹**：点击「添加文件夹」可多次添加，支持删除选中项
- **转码参数**：所有参数均为下拉框或数字输入框
- **开始转码**：点击后在底部日志框实时显示进度
- **输出位置**：每个输入文件夹下自动创建 `trans/` 子目录存放结果

### 自行打包
在 Windows 的 deface conda 环境中双击运行 `build.bat`，输出 `dist/transcode_gui.exe`。

---

## 命令行使用（transcode.py）

## 用法

```bash
python transcode.py <input> [options]
```

`input` 可以是单个文件、glob 模式或目录（自动扫描 mp4/mov/avi/mkv）。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | 必填 | 输入文件、glob 模式或目录 |
| `-o, --output` | 同输入目录 | 输出路径（文件或目录） |
| `--start-frame` | `0` | 起始帧 |
| `--end-frame` | `-1`（到结尾） | 结束帧 |
| `--fps` | 保持原始 | 输出帧率 |
| `--bitrate` | `1000` | 码率（kbps） |
| `--resolution` | `720p` | `480p` / `720p` / `1080p` / `4k` / `original` |
| `--codec` | `hevc` | `hevc` / `h264` / `vp9` / `av1` |
| `--preset` | `fast` | `ultrafast` → `slow`（速度/质量权衡，仅 CPU 编码器生效） |
| `--gpu` | `auto` | `auto` / `nvidia` / `amd` / `intel` / `cpu` |
| `--format` | `mp4` | 输出容器格式 |
| `--keep-audio` | 关闭（默认消音） | 加此标志保留音频，否则输出无声 |

## 分辨率对照

| 名称 | 分辨率 |
|------|--------|
| 480p | 854 × 480 |
| 720p | 1280 × 720 |
| 1080p | 1920 × 1080 |
| 4k | 3840 × 2160 |

## 示例

```bash
# 转为 1080p / 2000kbps / h264，默认消音
python transcode.py input.mp4 --resolution 1080p --bitrate 2000 --codec h264

# 保留音频
python transcode.py input.mp4 --keep-audio

# 批量处理目录，输出到 out/
python transcode.py ./videos/ -o ./out/ --resolution 720p

# 截取第 100~500 帧
python transcode.py input.mp4 --start-frame 100 --end-frame 500
```
