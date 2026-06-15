import os
# 在导入gradio前清除代理环境变量
for key in ['ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy']:
    os.environ.pop(key, None)

import sys
import re
import json
import threading
import urllib.parse
import gradio as gr
import subprocess
from pathlib import Path

# 日志框最多保留的行数（避免长视频日志无限增长拖慢界面）
MAX_LOG_LINES = 300

# 匹配 ANSI 转义控制码（tqdm 在非终端下用于光标定位，会污染日志）
ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

# 全局运行状态：与网页连接解耦。后台 reader 线程持续写入 RUN["states"]，
# 网页用 gr.Timer 每秒从这里读取渲染——这样网页关了/刷新/换浏览器都能 load 回正在跑的任务。
current_processes = []
RUN = {"active": False, "states": []}   # states: list[_ProcState]
RUN_LOCK = threading.Lock()

# 配置持久化：每次开始处理时存盘，页面加载时回填（连 app.py 重启也能恢复表单设置）
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "last_config.json")
CONFIG_KEYS = ["input_folder", "sfolder", "output_path", "detector", "thresh", "scale",
               "batchsize", "prefetch", "prep_workers", "prep_threads", "infer_threads",
               "bitrate_margin", "num_processes"]
CONFIG_DEFAULTS = {
    "input_folder": "", "sfolder": "", "output_path": "",
    "detector": "scrfd", "thresh": 0.5, "scale": "640x360",
    "batchsize": 64, "prefetch": 20, "prep_workers": 16, "prep_threads": 2,
    "infer_threads": 16, "bitrate_margin": 1.10, "num_processes": 4,
}

def _save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

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
    # 去掉空字节及其它不可见控制字符（部分拖拽源是 UTF-16，字符间夹 \x00，会导致
    # subprocess 报 "embedded null byte"）。\x00 去掉后 UTF-16 文本正好还原成正常路径。
    p = p.replace("\x00", "")
    p = "".join(ch for ch in p if ch == "\t" or ch >= " ")
    p = p.strip().strip('"').strip("'")
    if p.startswith("file://"):
        p = urllib.parse.unquote(urllib.parse.urlparse(p).path)
        # Windows 盘符形式 "/D:/..." -> "D:/..."
        if re.match(r'^/[A-Za-z]:', p):
            p = p[1:]
    return p

def stop_processing():
    global current_processes
    procs = current_processes
    if procs:
        n = len(procs)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        with RUN_LOCK:
            RUN["active"] = False
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
    with RUN_LOCK:
        RUN["active"] = False
        RUN["states"] = []
    try:
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
    except Exception:
        pass
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

def start_processing(input_path, sfolder, output_path, detector, thresh, scale,
                     batchsize, prefetch, prep_workers, prep_threads,
                     infer_threads, bitrate_margin, num_processes):
    """启动处理：拉起子进程 + 后台 reader 线程写入全局状态，然后立即返回。
    实时显示由 gr.Timer 从全局状态拉取，与本次点击/网页连接无关。"""
    global current_processes

    # 保存配置（用户输入的原始值），供刷新/重开/重启后回填
    cfg = dict(zip(CONFIG_KEYS, [input_path, sfolder, output_path, detector, thresh, scale,
                                 batchsize, prefetch, prep_workers, prep_threads,
                                 infer_threads, bitrate_margin, num_processes]))
    _save_config(cfg)

    # 固定默认值（界面已隐藏这些选项）
    replacewith = "mosaic"
    preset = "ultrafast"
    encoder = "libx264"

    input_path = clean_path(input_path)
    sfolder = clean_path(sfolder)
    output_path = clean_path(output_path)

    # 用 -u 关闭子进程缓冲，否则 tqdm/print 会被缓存、日志不实时
    if sfolder:
        base_cmd = [sys.executable, "-u", "deface.py", "--sfolder", sfolder]
    elif input_path:
        base_cmd = [sys.executable, "-u", "deface.py", input_path]
    else:
        return "", "", "请选择输入模式：普通文件夹或母文件夹"

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

    # 先停掉可能还在跑的旧任务，避免叠加
    for p in current_processes:
        try:
            p.terminate()
        except Exception:
            pass

    try:
        new_procs = []
        states = []
        for label, cmd in cmds:
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=os.path.dirname(__file__), env=env,
            )
            new_procs.append(p)
            st = _ProcState(label)
            st.lines.append(f"启动中：{' '.join(cmd)}")
            states.append(st)
            threading.Thread(target=_reader, args=(p, st), daemon=True).start()
        current_processes = new_procs
        with RUN_LOCK:
            RUN["active"] = True
            RUN["states"] = states
        return _render_global(states), _render_video(states), _render_log(states)
    except Exception as e:
        return "", "", f"错误: {str(e)}"

def tick():
    """gr.Timer 每秒调用：从全局状态渲染进度，与是谁的网页无关。"""
    with RUN_LOCK:
        states = list(RUN["states"])
    if not states:
        return "", "", ""
    if all(st.done for st in states):
        with RUN_LOCK:
            RUN["active"] = False
    return _render_global(states), _render_video(states), _render_log(states)

def load_state():
    """页面加载时：回填上次配置 + 接回正在跑的任务的当前进度。"""
    cfg = _load_config()

    def g(k):
        return cfg.get(k, CONFIG_DEFAULTS[k])

    with RUN_LOCK:
        states = list(RUN["states"])
    pf = _render_global(states) if states else ""
    pv = _render_video(states) if states else ""
    pl = _render_log(states) if states else ""
    return (g("input_folder"), g("sfolder"), g("output_path"), g("detector"),
            g("thresh"), g("scale"), g("batchsize"), g("prefetch"), g("prep_workers"),
            g("prep_threads"), g("infer_threads"), g("bitrate_margin"), g("num_processes"),
            pf, pv, pl)

with gr.Blocks(title="人脸脱敏工具v2.0") as demo:
    gr.Markdown("# 人脸脱敏工具 v2.0")

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

    # 所有输入组件（用于 demo.load 回填配置）
    _inputs = [input_folder, sfolder, output_path, detector, thresh, scale,
               batchsize, prefetch, prep_workers, prep_threads, infer_threads, bitrate_margin, num_processes]
    _progress = [folder_progress, video_progress, output_log]

    # 定时器：每秒从全局状态拉取进度并刷新（与是谁的网页无关，断线/刷新后自动接回）
    timer = gr.Timer(1.0)
    timer.tick(tick, None, _progress)

    run_btn.click(start_processing, _inputs, _progress)

    stop_btn.click(stop_processing, None, output_log)

    reset_btn.click(reset_all, None, _inputs + _progress)

    # 页面加载：回填上次配置 + 接回正在运行任务的进度
    demo.load(load_state, None, _inputs + _progress)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
