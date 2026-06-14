import os
# 在导入gradio前清除代理环境变量
for key in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(key, None)

import gradio as gr
import subprocess
from pathlib import Path

def process_videos(input_path, sfolder, output_path, detector, thresh, replacewith, scale, preset,
                   encoder, batchsize, prefetch, prep_workers, prep_threads,
                   infer_threads, bitrate_margin, progress=gr.Progress()):

    if not input_path:
        return "请选择输入文件夹"

    cmd = ["python", "deface.py", input_path]

    if output_path:
        cmd.extend(["--output", output_path])
    if sfolder:
        cmd.extend(["--sfolder", sfolder])
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
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, buffers=1, cwd=os.path.dirname(__file__))

        output = []
        for line in process.stdout:
            output.append(line)
            if "Processing" in line or "%" in line:
                progress(0.5, desc=line.strip())

        process.wait()
        return "\n".join(output) if output else "处理完成"

    except Exception as e:
        return f"错误: {str(e)}"

with gr.Blocks(title="人脸脱敏工具v1.0") as demo:
    gr.Markdown("# 人脸脱敏工具 v1.0")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 输入输出设置")
            input_folder = gr.Textbox(label="文件夹路径 (input)", placeholder="输入或拖动文件夹路径")
            sfolder = gr.Textbox(label="母文件夹路径 (sfolder)", placeholder="留空表示处理单个文件夹 (可拖动输入路径)")
            output_path = gr.Textbox(label="保存路径 (output)", placeholder="留空表示原路径 (可拖动输入路径)")

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

    gr.Markdown("### 处理进度")
    output_log = gr.Textbox(label="日志输出", lines=15, max_lines=20)

    run_btn.click(
        process_videos,
        [input_folder, sfolder, output_path, detector, thresh, replacewith, scale, preset,
         encoder, batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin],
        output_log
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
