import os
# 在导入gradio前清除代理环境变量
for key in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(key, None)

import sys
import re
import time
import threading
import urllib.parse
import gradio as gr
import subprocess
from pathlib import Path

# 日志框最多保留的行数（避免长视频日志无限增长拖慢界面）
MAX_LOG_LINES = 300

# 匹配 ANSI 转义控制码（tqdm 在非终端下用于光标定位，会污染日志）
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

# 全局变量存储当前所有子进程（多进程并行时为多个）
current_processes = []

def _bar_key(seg):
    # 是否是 tqdm 进度条行；是则返回其标识（用于区分批次条/帧条），否则 None
    if "%|" not in seg and "it/s" not in seg:
        return None
    m = re.match(r'\s*([^:|%]+):\s', seg)   # 形如 "Batch progress: ..." 的有描述进度条
    return m.group(1).strip() if m else "__bar__"

def _count_from_bar(seg):
    # 从进度条文本里抽 "x/y"（保留备用）
    m = re.search(r'(\d+)/(\d+)', seg or "")
    return f"{m.group(1)}/{m.group(2)}" if m else ""

class _ProcState:
    """单个子进程的输出状态：普通日志行 + 批次/帧进度条 + 当前文件 + 该分片视频总数。"""
    def __init__(self, label):
        self.label = label
        self.lines = []        # 普通日志行(str)，进度条不进这里
        self.folder = ""       # 批次进度条原文（Batch progress: x/y）
        self.video = ""        # 帧进度条原文
        self.current_file = "" # 当前正在处理的视频(带路径)
        self.total = 0         # 本进程(分片)负责的视频总数
        self.done = False      # 读取线程结束(进程退出)即为 True
        self.lock = threading.Lock()

    def add(self, seg):
        seg = ANSI_RE.sub("", seg).rstrip("\r\n")
        if seg.strip() == "":
            return
        key = _bar_key(seg)
        with self.lock:
            # "Input:" 可能被 tqdm 进度条(\r 刷新无换行)拼接到同一段里，
            # 所以无论该段是否是进度条，只要含 Input: 就提取当前文件并重置帧进度。
            if "Input:" in seg:
                self.current_file = seg.split("Input:", 1)[1].strip()
                self.video = ""
            if key == "Batch progress":
                self.folder = seg
            elif key is not None:
                self.video = seg
            else:
                # 视频总数：分片模式优先用 "[shard] ... handling M videos"，
                # 否则用 "[sfolder] Found N videos"（仅在分片数未知时）
                if "shard" in seg and "handling" in seg:
                    m = re.search(r'handling (\d+) videos', seg)
                    if m:
                        self.total = int(m.group(1))
                elif "Found" in seg and "video" in seg and not self.total:
                    m = re.search(r'Found (\d+) video', seg)
                    if m:
                        self.total = int(m.group(1))
                self.lines.append(seg)

def _reader(proc, st):
    # 逐字符读取，按 \r / \n 双分隔（兼容 tqdm 同行刷新）
    cur = ""
    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        if ch == "\n" or ch == "\r":
            st.add(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        st.add(cur)
    with st.lock:
        st.done = True

def _render_log(states):
    if len(states) == 1:
        st = states[0]
        with st.lock:
            return "\n".join(st.lines[-MAX_LOG_LINES:])
    # 多进程：每进程一块，带头部 + 末尾若干行
    per = max(8, MAX_LOG_LINES // len(states))
    blocks = []
    for st in states:
        with st.lock:
            tail = list(st.lines[-per:])
        blocks.append(f"─── {st.label} ───\n" + "\n".join(tail))
    return "\n\n".join(blocks)

def _render_global(states):
    # 全局聚合：运行中 | 已完成 | 剩余 (共 N 个视频)
    total = 0
    completed = 0
    running = 0
    known = False
    for st in states:
        with st.lock:
            t = st.total
            folder = st.folder
            cur = st.current_file
            done = st.done
        m = re.search(r'(\d+)/(\d+)', folder)
        x = int(m.group(1)) if m else 0
        y = int(m.group(2)) if m else 0
        if t:
            total += t
            known = True
        elif y:
            total += y
            known = True
        completed += x
        if (not done) and cur:
            running += 1
    if not known:
        return "准备中…"
    remaining = max(0, total - completed - running)
    return f"运行中 {running} | 已完成 {completed} | 剩余 {remaining}   (共 {total} 个视频)"

def _render_video(states):
    # 每个进程显示两行：当前文件(带路径) + 帧进度条(完整 tqdm 进度条)
    out = []
    for st in states:
        f = st.current_file or "-"
        bar = st.video or "-"
        if len(states) == 1:
            out.append(f"{f}\n{bar}")
        else:
            out.append(f"{st.label}: {f}\n  {bar}")
    return "\n".join(out)

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
    global current_processes
    if current_processes:
        n = len(current_processes)
        for p in current_processes:
            try:
                p.terminate()
            except Exception:
                pass
        return f"已发送停止信号（{n} 个进程）"
    return "没有正在运行的任务"

def reset_all():
    global current_processes
    for p in current_processes:
        try:
            p.terminate()
        except Exception:
            pass
    current_processes = []
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
        4,  # num_processes
        "",  # folder_progress
        "",  # video_progress
        "已重置所有设置"  # output_log
    )

def process_videos(input_path, sfolder, output_path, detector, thresh, scale,
                   batchsize, prefetch, prep_workers, prep_threads,
                   infer_threads, bitrate_margin, num_processes):
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
        base_cmd = [sys.executable, "-u", "deface.py", "--sfolder", sfolder]
    elif input_path:
        base_cmd = [sys.executable, "-u", "deface.py", input_path]
    else:
        yield "", "", "请选择输入模式：普通文件夹或母文件夹"
        return

    if output_path:
        base_cmd.extend(["--output", output_path])
    base_cmd.extend(["--detector", detector])
    base_cmd.extend(["--thresh", str(thresh)])
    base_cmd.extend(["--replacewith", replacewith])
    if scale and scale != "原尺寸":
        base_cmd.extend(["--scale", scale])
    base_cmd.extend(["--preset", preset])
    base_cmd.extend(["--encoder", encoder])
    base_cmd.extend(["--batchsize", str(batchsize)])
    base_cmd.extend(["--prefetch", str(prefetch)])
    base_cmd.extend(["--prep-workers", str(prep_workers)])
    base_cmd.extend(["--prep-threads", str(prep_threads)])
    base_cmd.extend(["--infer-threads", str(infer_threads)])
    base_cmd.extend(["--bitrate-margin", str(bitrate_margin)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # 多进程分片：仅 sfolder 母文件夹模式 + 进程数>1 时启用（按视频分片）
    n = max(1, int(num_processes))
    if n > 1 and sfolder:
        cmds = [(f"进程{i}", base_cmd + ["--num-shards", str(n), "--shard-id", str(i)])
                for i in range(n)]
    else:
        cmds = [("进程0", base_cmd)]

    try:
        global current_processes
        current_processes = []
        states = []
        threads = []
        for label, cmd in cmds:
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=os.path.dirname(__file__), env=env,
            )
            current_processes.append(p)
            st = _ProcState(label)
            st.lines.append(f"启动中：{' '.join(cmd)}")
            states.append(st)
            t = threading.Thread(target=_reader, args=(p, st), daemon=True)
            t.start()
            threads.append(t)

        # 周期性产出合并视图（多进程下避免逐字符 yield 过于频繁）
        while True:
            alive = any(t.is_alive() for t in threads)
            yield _render_global(states), _render_video(states), _render_log(states)
            if not alive:
                break
            time.sleep(0.3)

        for p in current_processes:
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        current_processes = []
        yield _render_global(states), _render_video(states), _render_log(states) or "处理完成"

    except Exception as e:
        current_processes = []
        yield "", "", f"错误: {str(e)}"

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
                    num_processes = gr.Slider(1, 8, value=4, step=1, label="并行进程数 (processes)",
                                              info="仅母文件夹(sfolder)模式生效：按视频分给多个进程并行，吃满多核+GPU。视频数≥进程数才有效")
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
            folder_progress = gr.Textbox(label="全局进度", value="", interactive=False)
            video_progress = gr.Textbox(label="当前视频进度", value="", interactive=False,
                                        lines=8, max_lines=18)
            output_log = gr.Textbox(label="日志输出", lines=28, max_lines=28, autoscroll=True)

    run_btn.click(
        process_videos,
        [input_folder, sfolder, output_path, detector, thresh, scale,
         batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin, num_processes],
        [folder_progress, video_progress, output_log]
    )

    stop_btn.click(stop_processing, None, output_log)

    reset_btn.click(
        reset_all,
        None,
        [input_folder, sfolder, output_path, detector, thresh, scale,
         batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin, num_processes,
         folder_progress, video_progress, output_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
