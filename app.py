import os
# 在导入gradio前清除代理环境变量
for key in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(key, None)

import sys
import re
import urllib.parse
import gradio as gr
import subprocess
from pathlib import Path

# 日志框最多保留的行数（避免长视频日志无限增长拖慢界面）
MAX_LOG_LINES = 300

# 匹配 ANSI 转义控制码（tqdm 在非终端下用于光标定位，会污染日志）
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

# 全局变量存储当前进程
current_process = None

def augment_cuda_path(env):
    """把 CUDA 及 conda 环境的 DLL 目录补进子进程 PATH（仅 Windows）。
    现象：终端 `python deface.py` 能跑，但 web 拉起的子进程报
    'CUDA_PATH is set but CUDA wasn't able to be loaded'。
    根因：onnxruntime-gpu 依赖的 cuDNN/cuBLAS 等 DLL 在 conda 环境的 Library\\bin 里，
    这些目录是 `conda activate` 才加进 PATH 的；若启动 web 时未激活，子进程就找不到。
    这里依据正在运行的解释器(sys.executable)推导 conda 环境目录，并补上 CUDA_PATH\\bin，
    相当于替子进程做一次 activate。非 Windows 直接返回。"""
    if os.name != "nt":
        return
    extra = []

    # 1) conda 环境自身的 DLL 目录（conda activate 会加这些）
    env_root = os.path.dirname(sys.executable)
    for sub in ("", r"Library\bin", r"Library\mingw-w64\bin", r"Library\usr\bin",
                "Scripts", "bin"):
        extra.append(os.path.join(env_root, sub) if sub else env_root)

    # 2) 系统 CUDA Toolkit 的 bin（CUDA_PATH / CUDA_PATH_V* 派生）
    for k, v in list(env.items()):
        if v and (k == "CUDA_PATH" or k.startswith("CUDA_PATH_")):
            extra.append(os.path.join(v, "bin"))
            extra.append(os.path.join(v, "bin", "x64"))

    cur = env.get("PATH", "")
    cur_lower = cur.lower()
    parts = []
    for p in extra:
        if os.path.isdir(p) and p.lower() not in cur_lower:
            parts.append(p)
    if parts:
        env["PATH"] = os.pathsep.join(parts) + os.pathsep + cur

def clean_path(p):
    """清理输入路径：去首尾空格/引号；若为浏览器拖拽的 file:// URL，则剥前缀并做 URL 解码。
    兼容 Windows 与 Linux：
      Linux  file:///media/x  -> /media/x   （保留开头斜杠）
      Windows file:///D:/x     -> D:/x       （去掉 urlparse 残留的前导斜杠）
    手动输入的普通路径原样保留，避免误伤路径中合法的 % 字符。"""
    if not p:
        return ""
    p = p.strip().strip('"').strip("'")
    if p.startswith("file://"):
        p = urllib.parse.unquote(urllib.parse.urlparse(p).path)
        # Windows 盘符形式 "/D:/..." -> "D:/..."
        if re.match(r'^/[A-Za-z]:', p):
            p = p[1:]
    return p

def stop_processing():
    global current_process
    if current_process:
        current_process.terminate()
        return "已发送停止信号"
    return "没有正在运行的任务"

def reset_all():
    global current_process
    if current_process:
        current_process.terminate()
        current_process = None
    # 返回所有组件的默认值
    return (
        "",  # input_folder
        "",  # sfolder
        "",  # output_path
        "scrfd",  # detector
        0.5,  # thresh
        "640x360",  # scale
        64,  # batchsize
        20,  # prefetch
        16,  # prep_workers
        2,  # prep_threads
        16,  # infer_threads
        1.10,  # bitrate_margin
        "",  # folder_progress
        "",  # video_progress
        "已重置所有设置"  # output_log
    )

def process_videos(input_path, sfolder, output_path, detector, thresh, scale,
                   batchsize, prefetch, prep_workers, prep_threads,
                   infer_threads, bitrate_margin):
    # 固定默认值（界面已隐藏这些选项）
    replacewith = "mosaic"
    preset = "ultrafast"
    encoder = "libx264"

    # 清理路径：去首尾空格，处理 file:// 拖拽前缀并做 URL 解码（中文/空格会被百分号编码）
    input_path = clean_path(input_path)
    sfolder = clean_path(sfolder)
    output_path = clean_path(output_path)

    # 二选一：sfolder模式或普通input模式
    # 用 -u 关闭子进程缓冲，否则 tqdm/print 会被缓存、日志不实时
    if sfolder:
        cmd = [sys.executable, "-u", "deface.py", "--sfolder", sfolder]
    elif input_path:
        cmd = [sys.executable, "-u", "deface.py", input_path]
    else:
        yield "", "", "请选择输入模式：普通文件夹或母文件夹"
        return

    if output_path:
        cmd.extend(["--output", output_path])
    cmd.extend(["--detector", detector])
    cmd.extend(["--thresh", str(thresh)])
    cmd.extend(["--replacewith", replacewith])
    if scale and scale != "原尺寸":
        cmd.extend(["--scale", scale])
    cmd.extend(["--preset", preset])
    cmd.extend(["--encoder", encoder])
    cmd.extend(["--batchsize", str(batchsize)])
    cmd.extend(["--prefetch", str(prefetch)])
    cmd.extend(["--prep-workers", str(prep_workers)])
    cmd.extend(["--prep-threads", str(prep_threads)])
    cmd.extend(["--infer-threads", str(infer_threads)])
    cmd.extend(["--bitrate-margin", str(bitrate_margin)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    augment_cuda_path(env)

    folder_prog = ""
    video_prog = ""
    lines = []   # 每项为 (bar_key, 文本)；bar_key 为 None 表示普通日志行

    def render():
        return "\n".join(t for _, t in lines[-MAX_LOG_LINES:])

    def bar_key(seg):
        # 是否是 tqdm 进度条行；是则返回其标识（用于判断"同一个条"），否则 None
        if "%|" not in seg and "it/s" not in seg:
            return None
        m = re.match(r'\s*([^:|%]+):\s', seg)   # 形如 "Batch progress: ..." 的有描述进度条
        return m.group(1).strip() if m else "__bar__"

    def add_segment(seg):
        # 处理一个以 \r 或 \n 分隔出的片段：去掉 ANSI 码。
        # tqdm 进度条 -> 只更新右侧独立进度框（天生原地刷新）；普通日志 -> 追加到日志框
        nonlocal folder_prog, video_prog
        seg = ANSI_RE.sub("", seg).rstrip("\r\n")
        if seg.strip() == "":
            return
        key = bar_key(seg)
        if key == "Batch progress":
            folder_prog = seg
        elif key is not None:
            video_prog = seg
        else:
            lines.append((None, seg))

    try:
        global current_process
        current_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=os.path.dirname(__file__), env=env,
        )

        lines.append((None, f"启动中：{' '.join(cmd)}"))
        yield folder_prog, video_prog, render()

        cur = ""
        while True:
            ch = current_process.stdout.read(1)
            if ch == "":
                break
            if ch == "\n" or ch == "\r":
                add_segment(cur)
                cur = ""
                yield folder_prog, video_prog, render()
            else:
                cur += ch

        if cur:
            add_segment(cur)

        current_process.wait()
        current_process = None
        yield folder_prog, video_prog, render() or "处理完成"

    except Exception as e:
        current_process = None
        yield folder_prog, video_prog, f"错误: {str(e)}"

with gr.Blocks(title="人脸脱敏工具v1.0") as demo:
    gr.Markdown("# 人脸脱敏工具 v1.0")

    with gr.Row():
        # 左列：所有设置参数 + 操作按钮
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 输入输出设置")
                    gr.Markdown('<p style="color: gray; font-size: 0.9em; margin-top: -10px;">可拖动输入路径 | 以下两种模式二选一</p>')
                    input_folder = gr.Textbox(label="文件夹路径 (input)", placeholder="处理单个文件夹中的所有视频")
                    sfolder = gr.Textbox(label="母文件夹路径 (sfolder)", placeholder="处理所有子文件夹中的视频")
                    output_path = gr.Textbox(label="保存路径 (output)", placeholder="留空表示原路径")

                    gr.Markdown("### 检测参数")
                    detector = gr.Radio(["centerface", "scrfd"], value="scrfd", label="检测器 (detector)")
                    thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="检测阈值 (thresh)")

                    gr.Markdown("### 处理参数")
                    scale = gr.Dropdown(["原尺寸", "640x360", "1280x720"], value="640x360",
                                       label="缩放尺寸 (scale)")

                with gr.Column():
                    gr.Markdown("### 性能参数")
                    batchsize = gr.Slider(1, 128, value=64, step=1, label="批处理大小 (batchsize)")
                    prefetch = gr.Slider(1, 50, value=20, step=1, label="预取帧数 (prefetch)")
                    prep_workers = gr.Slider(1, 32, value=16, step=1, label="预处理进程数 (prep-workers)")
                    prep_threads = gr.Slider(1, 16, value=2, step=1, label="预处理线程数 (prep-threads)")
                    infer_threads = gr.Slider(1, 32, value=16, step=1, label="推理线程数 (infer-threads)")
                    bitrate_margin = gr.Slider(1.0, 2.0, value=1.10, step=0.05,
                                              label="码率余量 (bitrate-margin)")

            run_btn = gr.Button("开始处理", variant="primary", size="lg")
            with gr.Row():
                stop_btn = gr.Button("停止处理", variant="stop", size="lg")
                reset_btn = gr.Button("一键重置", variant="secondary", size="lg")

        # 右列：进度与日志，单独成栏，无需滚动即可看到
        with gr.Column(scale=2):
            gr.Markdown("### 处理进度")
            folder_progress = gr.Textbox(label="文件夹进度 (批次)", value="", interactive=False)
            video_progress = gr.Textbox(label="当前视频进度", value="", interactive=False)
            output_log = gr.Textbox(label="日志输出", lines=28, max_lines=28, autoscroll=True)

    run_btn.click(
        process_videos,
        [input_folder, sfolder, output_path, detector, thresh, scale,
         batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin],
        [folder_progress, video_progress, output_log]
    )

    stop_btn.click(stop_processing, None, output_log)

    reset_btn.click(
        reset_all,
        None,
        [input_folder, sfolder, output_path, detector, thresh, scale,
         batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin,
         folder_progress, video_progress, output_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
