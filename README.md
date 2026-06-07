# Deface - 视频人脸匿名化工具

基于ONNX Runtime的高性能视频人脸检测与匿名化工具，支持多GPU并行加速。

## 🚀 新增功能

### 多GPU并行支持
- **自动检测GPU数量**：支持pynvml、PyTorch、ONNX Runtime多种检测方式
- **独立队列架构**：每个GPU拥有专属推理队列，消除竞争
- **负载均衡**：Round-robin分配策略确保GPU负载均衡
- **环境变量支持**：尊重`CUDA_VISIBLE_DEVICES`配置

### 性能分析增强
- **详细计时统计**：blob创建、ONNX推理、结果解码的细分计时
- **多GPU统计聚合**：自动汇总所有GPU的性能数据
- **实时性能监控**：每10秒输出详细的性能分析

### 用户体验改进
- **信号处理**：Ctrl+C退出时自动清理OpenCV窗口，避免终端乱码
- **调试信息**：可选的batch shape验证信息

## 📦 安装

```bash
pip install -r requirements.txt
```

**核心依赖**：
- `onnxruntime-gpu` - GPU推理引擎
- `nvidia-ml-py3` - GPU检测和管理
- `opencv-python` - 图像处理
- `numpy`, `imageio`, `tqdm` - 基础工具

## ⚙️ 参数设置指南

### 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batchsize` | 8 | 批处理大小，影响GPU利用率和显存占用 |
| `--prefetch` | 2 | 队列预取深度，影响数据流水线效率 |
| `--prep-workers` | 6 | 预处理并行worker数量 |
| `--prep-threads` | 2 | 每个worker的线程数 |
| `--infer-threads` | 4 | 推理worker的线程数（blob创建和解码） |
| `--preset` | fast | 编码器预设（ultrafast/fast/medium/slow） |
| `--encoder` | auto | 视频编码器（auto/libx264/h264_nvenc等） |
| `--scale` | None | 推理分辨率（如640x360，降低可提速） |

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--thresh` | 0.2 | 检测阈值（降低减少漏检，提高减少误检） |
| `--mask-scale` | 1.3 | 人脸遮罩缩放因子 |
| `--replacewith` | blur | 匿名化方式（blur/mosaic/solid/img/none） |

## 🎯 场景推荐配置

### 场景1：快速处理（牺牲部分质量）
适用于：快速预览、测试用途

```bash
python deface.py input.mp4 \
  --preset ultrafast \
  --batchsize 64 \
  --prefetch 16 \
  --prep-workers 12 \
  --infer-threads 6 \
  --scale 640x360 \
  --encoder h264_nvenc
```

**预期速度**：200+ fps（双GPU）

### 场景2：平衡模式（推荐）
适用于：日常使用，质量与速度平衡

```bash
python deface.py input.mp4 \
  --preset fast \
  --batchsize 32 \
  --prefetch 8 \
  --prep-workers 12 \
  --infer-threads 4 \
  --scale 640x360
```

**预期速度**：100-150 fps（双GPU）

### 场景3：高质量输出
适用于：最终交付、重要视频

```bash
python deface.py input.mp4 \
  --preset slow \
  --batchsize 16 \
  --prefetch 4 \
  --thresh 0.15 \
  --bitrate-margin 1.8
```

**预期速度**：40-60 fps（双GPU）

### 场景4：显存受限环境
适用于：GPU显存较小（<6GB）

```bash
python deface.py input.mp4 \
  --batchsize 16 \
  --prefetch 4 \
  --prep-workers 6 \
  --infer-threads 2 \
  --scale 640x360
```

## 🔧 多GPU使用

### 自动检测所有GPU
```bash
python deface.py input.mp4 --batchsize 32
# 自动检测并使用所有可用GPU
```

### 指定使用特定GPU
```bash
# 只使用GPU 0
CUDA_VISIBLE_DEVICES=0 python deface.py input.mp4

# 使用GPU 0和1
CUDA_VISIBLE_DEVICES=0,1 python deface.py input.mp4

# 使用GPU 1和2
CUDA_VISIBLE_DEVICES=1,2 python deface.py input.mp4
```

### 多GPU性能调优
**关键原则**：
- **增大batchsize**：多GPU需要更大的批次才能充分利用（建议32-64）
- **增加prefetch**：避免GPU等待数据（建议8-16）
- **增加prep-workers**：确保数据预处理不成为瓶颈（建议12-16）

## 📊 性能分析解读

运行时每10秒会输出详细的性能分析：

```
[speed] 100.59 fps avg, 255.94 fps recent, 4864/133951 frames
[inference] blob=19.1% infer=153.6% decode=2.3% | proc=0.5% qget=18.6% qput=0.2% other=-93.7%
```

### 关键指标含义

| 指标 | 含义 | 优化建议 |
|------|------|----------|
| **blob** | ONNX输入blob创建时间 | 如果>20%，增加`--infer-threads` |
| **infer** | ONNX模型推理时间 | >100%说明多GPU并行正常工作 |
| **decode** | 检测结果解码时间 | 如果>15%，增加`--infer-threads` |
| **qget** | 队列等待时间 | 如果>15%，增加`--prefetch`和`--prep-workers` |
| **proc** | 后处理时间 | 通常很小，不是瓶颈 |

### 性能瓶颈诊断

**瓶颈1：qget过高（>15%）**
- 现象：GPU在等待数据
- 解决：增加`--prefetch 16`、`--prep-workers 16`

**瓶颈2：blob过高（>20%）**
- 现象：blob创建慢
- 解决：增加`--infer-threads 8`

**瓶颈3：队列经常为0**
- 现象：上游处理太慢
- 解决：增加`--prep-workers`、降低`--scale`

## ❓ 常见问题

### Q1: 显存不足错误
```
RuntimeException: Failed to allocate memory for requested buffer
```

**解决方案**：
1. 降低`--batchsize`（从32降到16或8）
2. 降低`--prefetch`（从16降到4）
3. 使用更小的`--scale`（如320x180）
4. 只使用单GPU：`CUDA_VISIBLE_DEVICES=0`

### Q2: Ctrl+C退出后终端乱码
**解决方案**：已修复！代码已添加信号处理器自动清理。
如仍遇到，手动输入：`reset`或`stty sane`

### Q3: 多GPU不生效
**检查方法**：
1. 查看日志是否显示：`[multi-gpu] Enabled - using N GPU(s)`
2. 查看`[inference]`中infer是否>100%（说明并行工作）

**解决方案**：
- 确保安装了`nvidia-ml-py3`：`pip install nvidia-ml-py3`
- 检查CUDA_VISIBLE_DEVICES设置

## 📝 更新日志

### 2026-06-07
- ✨ 添加多GPU自动检测和并行支持
- ✨ 独立队列架构消除GPU竞争
- ✨ 详细的inference内部性能分析
- 🐛 修复Ctrl+C退出终端乱码问题
- 📦 添加nvidia-ml-py3依赖

