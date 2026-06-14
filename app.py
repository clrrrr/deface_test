import os
# 在导入gradio前清除代理环境变量
for key in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(key, None)

import gradio as gr
import subprocess
from pathlib import Path
from urllib.parse import unquote

# 全局变量存储当前进程
current_process = None

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
        "mosaic",  # replacewith
        "640x360",  # scale
        "ultrafast",  # preset
        "libx264",  # encoder
        64,  # batchsize
        20,  # prefetch
        16,  # prep_workers
        2,  # prep_threads
        16,  # infer_threads
        1.10,  # bitrate_margin
        "已重置所有设置",  # status_text
        "",  # folder_progress
        "",  # video_progress
        ""  # output_log
    )

def process_videos(input_path, sfolder, output_path, detector, thresh, replacewith, scale, preset,
                   encoder, batchsize, prefetch, prep_workers, prep_threads,
                   infer_threads, bitrate_margin):

    # 清理路径首尾空格和file://前缀
    input_path = input_path.strip() if input_path else ""
    sfolder = sfolder.strip() if sfolder else ""
    output_path = output_path.strip() if output_path else ""

    if input_path.startswith("file://"):
        input_path = input_path[7:]
    if sfolder.startswith("file://"):
        sfolder = sfolder[7:]
    if output_path.startswith("file://"):
        output_path = output_path[7:]

    # URL解码（处理中文路径）
    input_path = unquote(input_path)
    sfolder = unquote(sfolder)
    output_path = unquote(output_path)

    # 二选一：sfolder模式或普通input模式
    if sfolder:
        cmd = ["python", "deface.py", "--sfolder", sfolder]
    elif input_path:
        cmd = ["python", "deface.py", input_path]
    else:
        return "请选择输入模式：普通文件夹或母文件夹"

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

    try:
        global current_process
        current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, cwd=os.path.dirname(__file__))

        output = []
        folder_prog = ""
        video_prog = ""

        for line in current_process.stdout:
            output.append(line)
            # 解析进度信息
            if "folder" in line.lower() or "subfolder" in line.lower():
                import re
                match = re.search(r'(\d+)/(\d+)', line)
                if match:
                    folder_prog = f"{match.group(1)}/{match.group(2)}"
            if "video" in line.lower() or "processing" in line.lower():
                import re
                match = re.search(r'(\d+)/(\d+)', line)
                if match:
                    video_prog = f"{match.group(1)}/{match.group(2)}"

        current_process.wait()
        current_process = None
        log = "\n".join(output) if output else "处理完成"
        return folder_prog, video_prog, log

    except Exception as e:
        current_process = None
        return "", "", f"错误: {str(e)}"

with gr.Blocks(title="人脸脱敏工具v1.0") as demo:
    gr.Markdown("# 人脸脱敏工具 v1.0")

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
            replacewith = gr.Radio(["blur", "solid", "none", "mosaic"], value="mosaic",
                                   label="替换方式 (replacewith)")
            scale = gr.Dropdown(["原尺寸", "640x360", "1280x720"], value="640x360",
                               label="缩放尺寸 (scale)")

        with gr.Column():
            gr.Markdown("### 性能参数")
            preset = gr.Dropdown(["ultrafast", "fast", "medium", "slow"], value="ultrafast",
                                label="编码预设 (preset)")
            encoder = gr.Textbox(value="libx264", label="编码器 (encoder)")
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

    gr.Markdown("### 处理进度")
    status_text = gr.Textbox(label="状态", value="", interactive=False)
    folder_progress = gr.Textbox(label="文件夹进度", value="", visible=False, interactive=False)
    video_progress = gr.Textbox(label="当前文件夹内进度", value="", interactive=False)
    output_log = gr.Textbox(label="日志输出", lines=15, max_lines=20)

    run_btn.click(
        process_videos,
        [input_folder, sfolder, output_path, detector, thresh, replacewith, scale, preset,
         encoder, batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin],
        [folder_progress, video_progress, output_log]
    )

    stop_btn.click(stop_processing, None, status_text)

    reset_btn.click(
        reset_all,
        None,
        [input_folder, sfolder, output_path, detector, thresh, replacewith, scale, preset,
         encoder, batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin,
         status_text, folder_progress, video_progress, output_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
