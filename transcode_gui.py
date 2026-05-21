#!/usr/bin/env python3
import sys, os, glob, types, threading, io, re, time, json
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import transcode

VIDEO_EXTS = ('mp4', 'MP4', 'mov', 'MOV', 'avi', 'AVI', 'mkv', 'MKV')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')
LOG_FILE = '.transcode_log.json'


def fmt_time(seconds):
    if not (0 <= seconds < float('inf')):
        return '--:--'
    s = int(seconds)
    return f'{s//60:02d}:{s%60:02d}'


# ── log 读写 ────────────────────────────────────────────────────

def read_log(out_dir):
    path = os.path.join(out_dir, LOG_FILE)
    if not os.path.exists(path):
        return {'done': [], 'interrupted': None}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'done': [], 'interrupted': None}


def write_log(out_dir, done, interrupted=None):
    path = os.path.join(out_dir, LOG_FILE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'done': done, 'interrupted': interrupted}, f, ensure_ascii=False)


# ── stdout 重定向 ───────────────────────────────────────────────

class TextRedirector(io.TextIOBase):
    def __init__(self, widget):
        self.widget = widget

    def write(self, s):
        s = ANSI_RE.sub('', s)
        if s:
            self.widget.after(0, self._append, s)
        return len(s)

    def flush(self):
        pass

    def _append(self, s):
        self.widget.configure(state='normal')
        self.widget.insert('end', s)
        self.widget.see('end')
        self.widget.configure(state='disabled')


# ── 主窗口 ──────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Video Transcoder')
        self.minsize(720, 600)
        self._stop_event = threading.Event()
        self._total_start = self._file_start = self._last_ui_update = 0.0
        self._build_folder_panel()
        self._build_params_panel()
        self._build_status_panel()
        self._build_bottom_panel()
        self.folder_list.bind('<<ListboxSelect>>', lambda e: self._refresh_status())

    # ── 文件夹面板 ──────────────────────────────────────────────
    def _build_folder_panel(self):
        frm = ttk.LabelFrame(self, text='输入文件夹')
        frm.pack(fill='both', expand=True, padx=8, pady=4)

        self.folder_list = tk.Listbox(frm, selectmode='extended', height=5)
        self.folder_list.pack(side='left', fill='both', expand=True)

        sb = ttk.Scrollbar(frm, orient='vertical', command=self.folder_list.yview)
        sb.pack(side='left', fill='y')
        self.folder_list.configure(yscrollcommand=sb.set)

        btn_frm = ttk.Frame(frm)
        btn_frm.pack(side='left', padx=4)
        ttk.Button(btn_frm, text='添加文件夹',   command=self.add_folder).pack(pady=2, fill='x')
        ttk.Button(btn_frm, text='添加母文件夹', command=self.add_parent_folder).pack(pady=2, fill='x')
        ttk.Button(btn_frm, text='删除选中',     command=self.remove_selected).pack(pady=2, fill='x')

    # ── 参数面板 ────────────────────────────────────────────────
    def _build_params_panel(self):
        frm = ttk.LabelFrame(self, text='转码参数')
        frm.pack(fill='x', padx=8, pady=4)

        self.vars = {}
        dropdowns = [
            ('resolution', '分辨率',  ['720p', '480p', '1080p', '4k', 'original']),
            ('codec',      '编码器',  ['hevc', 'h264', 'vp9', 'av1']),
            ('preset',     '预设',    ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow']),
            ('gpu',        'GPU',     ['auto', 'nvidia', 'amd', 'intel', 'cpu']),
            ('fmt',        '格式',    ['mp4', 'mov', 'mkv', 'avi']),
        ]
        for col, (key, label, choices) in enumerate(dropdowns):
            ttk.Label(frm, text=label).grid(row=0, column=col*2, padx=4, pady=6, sticky='e')
            v = tk.StringVar(value=choices[0])
            self.vars[key] = v
            ttk.Combobox(frm, textvariable=v, values=choices, width=11, state='readonly') \
                .grid(row=0, column=col*2+1, padx=4)

        entries = [
            ('bitrate',     '码率(kbps)',     '1000'),
            ('start_frame', '起始帧',         '0'),
            ('end_frame',   '结束帧(空=末尾)', ''),
            ('fps',         'FPS(空=原)',      ''),
        ]
        for col, (key, label, default) in enumerate(entries):
            ttk.Label(frm, text=label).grid(row=1, column=col*2, padx=4, pady=6, sticky='e')
            v = tk.StringVar(value=default)
            self.vars[key] = v
            ttk.Entry(frm, textvariable=v, width=11).grid(row=1, column=col*2+1, padx=4)

        self.vars['keep_audio'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='保留音频', variable=self.vars['keep_audio']) \
            .grid(row=1, column=len(entries)*2, padx=8)

    # ── 状态信息面板 ────────────────────────────────────────────
    def _build_status_panel(self):
        frm = ttk.LabelFrame(self, text='进度状态')
        frm.pack(fill='x', padx=8, pady=2)
        self.status_text = tk.Text(frm, height=3, state='disabled',
                                   background=self.cget('background'), relief='flat',
                                   font=('TkDefaultFont', 9), wrap='word')
        self.status_text.pack(fill='x', padx=4, pady=2)
        self.status_text.tag_configure('resume', foreground='#0066cc')
        self.status_text.tag_configure('interrupted', foreground='#cc6600')
        self.status_text.tag_configure('normal', foreground='gray')

    def _set_status(self, lines):
        """lines: list of (text, tag)"""
        self.after(0, self._apply_status, lines)

    def _apply_status(self, lines):
        self.status_text.configure(state='normal')
        self.status_text.delete('1.0', 'end')
        for text, tag in lines:
            self.status_text.insert('end', text + '\n', tag)
        self.status_text.configure(state='disabled')

    def _refresh_status(self):
        """文件夹列表变化时扫描 log 并更新状态面板"""
        folders = list(self.folder_list.get(0, 'end'))
        lines = []
        for folder in folders:
            out_dir = os.path.join(folder, 'trans')
            log = read_log(out_dir)
            name = os.path.basename(folder)
            if log['interrupted']:
                lines.append((f'[{name}] 上次中断于: {log["interrupted"]}，将重新转码该文件', 'interrupted'))
            elif log['done']:
                lines.append((f'[{name}] 读取到上次进度，已完成 {len(log["done"])} 个文件，将继续处理剩余文件', 'resume'))
        if not lines:
            lines = [('暂无历史进度记录', 'normal')]
        self._apply_status(lines)

    # ── 底部面板 ────────────────────────────────────────────────
    def _build_bottom_panel(self):
        btn_frm = ttk.Frame(self)
        btn_frm.pack(pady=4)
        self.start_btn = ttk.Button(btn_frm, text='开始转码', command=self.start_transcode)
        self.start_btn.pack(side='left', padx=4)
        self.stop_btn = ttk.Button(btn_frm, text='停止', command=self.stop_transcode, state='disabled')
        self.stop_btn.pack(side='left', padx=4)

        prog_frm = ttk.Frame(self)
        prog_frm.pack(fill='x', padx=8)

        self.total_label = ttk.Label(prog_frm, text='总进度: 0/0', width=42, anchor='w')
        self.total_label.grid(row=0, column=0, sticky='w', pady=2)
        self.total_bar = ttk.Progressbar(prog_frm, mode='determinate')
        self.total_bar.grid(row=0, column=1, sticky='ew', padx=8)

        self.file_label = ttk.Label(prog_frm, text='当前文件: -', width=42, anchor='w')
        self.file_label.grid(row=1, column=0, sticky='w', pady=2)
        self.file_bar = ttk.Progressbar(prog_frm, mode='determinate')
        self.file_bar.grid(row=1, column=1, sticky='ew', padx=8)

        self.current_file_label = ttk.Label(prog_frm, text='', anchor='w', foreground='gray')
        self.current_file_label.grid(row=2, column=0, columnspan=2, sticky='ew', padx=2)

        prog_frm.columnconfigure(1, weight=1)

        self.log_text = scrolledtext.ScrolledText(self, height=8, state='disabled')
        self.log_text.pack(fill='both', expand=True, padx=8, pady=4)

    # ── 操作 ────────────────────────────────────────────────────
    def add_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.folder_list.insert('end', d)
            self._refresh_status()

    def add_parent_folder(self):
        d = filedialog.askdirectory(title='选择母文件夹')
        if not d:
            return
        existing = set(self.folder_list.get(0, 'end'))
        for entry in sorted(os.scandir(d), key=lambda e: e.name):
            if entry.is_dir() and entry.path not in existing:
                self.folder_list.insert('end', entry.path)
        self._refresh_status()

    def remove_selected(self):
        for i in reversed(self.folder_list.curselection()):
            self.folder_list.delete(i)
        self._refresh_status()

    def start_transcode(self):
        folders = list(self.folder_list.get(0, 'end'))
        if not folders:
            return
        self._stop_event.clear()
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        params = {k: v.get() for k, v in self.vars.items()}
        threading.Thread(target=self._worker, args=(folders, params), daemon=True).start()

    def stop_transcode(self):
        self._stop_event.set()
        self.stop_btn.configure(state='disabled')

    # ── 进度更新 ────────────────────────────────────────────────
    def _set_total(self, cur, total, name=''):
        elapsed = time.time() - self._total_start
        eta = (elapsed / cur * (total - cur)) if cur > 0 else float('inf')
        text = f'总进度: {cur}/{total}  已用:{fmt_time(elapsed)}  剩余:{fmt_time(eta)}'
        self.after(0, lambda: (
            self.total_bar.configure(maximum=max(total, 1), value=cur),
            self.total_label.configure(text=text),
            self.current_file_label.configure(text=name),
        ))

    def _set_file(self, cur, total):
        now = time.time()
        if cur < total and now - self._last_ui_update < 0.2:
            return
        self._last_ui_update = now
        elapsed = time.time() - self._file_start
        eta = (elapsed / cur * (total - cur)) if cur > 0 else float('inf')
        text = f'当前文件: {cur}/{total}帧  已用:{fmt_time(elapsed)}  剩余:{fmt_time(eta)}'
        self.after(0, lambda: (
            self.file_bar.configure(maximum=max(total, 1), value=cur),
            self.file_label.configure(text=text),
        ))

    # ── 工作线程 ────────────────────────────────────────────────
    def _worker(self, folders, params):
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = TextRedirector(self.log_text)
        try:
            gpu = transcode.detect_gpu() if params['gpu'] == 'auto' else params['gpu']
            encoder = (transcode.ENCODERS.get((params['codec'], gpu))
                       or transcode.ENCODERS.get((params['codec'], 'cpu')))
            print(f"GPU: {gpu}  Encoder: {encoder}\n")

            args = types.SimpleNamespace(
                resolution  = params['resolution'],
                codec       = params['codec'],
                preset      = params['preset'],
                fmt         = params['fmt'],
                bitrate     = int(params['bitrate'] or 1000),
                start_frame = int(params['start_frame'] or 0),
                end_frame   = int(params['end_frame']) if params['end_frame'] else -1,
                fps         = float(params['fps']) if params['fps'] else None,
                keep_audio  = bool(params['keep_audio']),
                output      = None,
            )

            # 收集所有文件（去重）
            all_files = []
            seen = set()
            for folder in folders:
                for ext in VIDEO_EXTS:
                    for f in glob.glob(os.path.join(folder, f'*.{ext}')):
                        key = f.lower()
                        if key not in seen:
                            seen.add(key)
                            all_files.append((folder, f))

            # 按文件夹分组，读取各自 log，跳过已完成文件
            pending = []
            for folder, f in all_files:
                out_dir = os.path.join(folder, 'trans')
                log = read_log(out_dir)
                name = os.path.basename(f)
                if name in log['done']:
                    continue  # 已完成，跳过
                # 中断中的文件：删掉不完整的输出，重新转
                if name == log['interrupted']:
                    stem = os.path.splitext(name)[0]
                    # 匹配所有可能的输出文件名（不同 suffix/格式）
                    for candidate in glob.glob(os.path.join(out_dir, f'{stem}_*')):
                        try:
                            os.remove(candidate)
                        except Exception:
                            pass
                pending.append((folder, f))

            total = len(pending)
            self._total_start = time.time()
            self._set_total(0, total)

            # 按文件夹维护各自的 done 列表
            folder_done = {}
            for folder, _ in pending:
                out_dir = os.path.join(folder, 'trans')
                if folder not in folder_done:
                    folder_done[folder] = list(read_log(out_dir)['done'])

            for idx, (folder, f) in enumerate(pending):
                if self._stop_event.is_set():
                    # 记录中断状态
                    name = os.path.basename(f)
                    out_dir = os.path.join(folder, 'trans')
                    write_log(out_dir, folder_done[folder], interrupted=name)
                    self._set_status([(f'已停止，{name} 被中断，下次将从此文件继续', 'interrupted')])
                    print(f"\n已停止，中断于: {name}")
                    break

                out_dir = os.path.join(folder, 'trans')
                os.makedirs(out_dir, exist_ok=True)
                args.output = out_dir

                name = os.path.basename(f)
                # 标记为正在处理
                write_log(out_dir, folder_done[folder], interrupted=name)
                self._set_status([(f'正在处理: {name}', 'normal')])

                self._set_total(idx, total, name)
                self._file_start = time.time()
                self._set_file(0, 1)
                print(f"\n[{idx+1}/{total}] {f}")
                try:
                    rc = transcode.process_file(f, args, encoder,
                                               progress_cb=self._set_file,
                                               stop_event=self._stop_event)
                    if rc == 0:
                        # 成功完成，记入 done，清除 interrupted
                        folder_done[folder].append(name)
                        write_log(out_dir, folder_done[folder], interrupted=None)
                    # rc != 0 且非异常 = 被 stop_event 中断，interrupted 已在开始前写好，保持不变
                except Exception as e:
                    print(f"  跳过: {e}")

                self._set_total(idx + 1, total, name)

            else:
                print("\n全部完成。")
                # 清理所有 log（用原始 folders 列表，不依赖 pending）
                for done_folder in folders:
                    out_dir = os.path.join(done_folder, 'trans')
                    log_path = os.path.join(out_dir, LOG_FILE)
                    try:
                        os.remove(log_path)
                    except Exception:
                        pass
                self._set_status([('全部完成', 'normal')])

            self.after(0, lambda: self.file_label.configure(text='当前文件: -'))
            self.after(0, lambda: self.file_bar.configure(value=0))
            self._refresh_status()
        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self.after(0, lambda: (
                self.start_btn.configure(state='normal'),
                self.stop_btn.configure(state='disabled'),
            ))


if __name__ == '__main__':
    App().mainloop()
