#!/usr/bin/env python3
import sys, os, glob, types, threading, io, re, time
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import transcode

VIDEO_EXTS = ('mp4', 'MP4', 'mov', 'MOV', 'avi', 'AVI', 'mkv', 'MKV')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')


def fmt_time(seconds):
    if not (0 <= seconds < float('inf')):
        return '--:--'
    s = int(seconds)
    return f'{s//60:02d}:{s%60:02d}'


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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Video Transcoder')
        self.minsize(720, 560)
        self._stop_event = threading.Event()
        self._total_start = self._file_start = 0.0
        self._build_folder_panel()
        self._build_params_panel()
        self._build_bottom_panel()

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

        prog_frm.columnconfigure(1, weight=1)

        self.log_text = scrolledtext.ScrolledText(self, height=10, state='disabled')
        self.log_text.pack(fill='both', expand=True, padx=8, pady=4)

    # ── 操作 ────────────────────────────────────────────────────
    def add_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.folder_list.insert('end', d)

    def add_parent_folder(self):
        d = filedialog.askdirectory(title='选择母文件夹')
        if not d:
            return
        existing = set(self.folder_list.get(0, 'end'))
        for entry in sorted(os.scandir(d), key=lambda e: e.name):
            if entry.is_dir() and entry.path not in existing:
                self.folder_list.insert('end', entry.path)

    def remove_selected(self):
        for i in reversed(self.folder_list.curselection()):
            self.folder_list.delete(i)

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
        if name:
            text += f'  {name}'
        self.after(0, lambda: (
            self.total_bar.configure(maximum=max(total, 1), value=cur),
            self.total_label.configure(text=text),
        ))

    def _set_file(self, cur, total):
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

            all_files = []
            for folder in folders:
                for ext in VIDEO_EXTS:
                    all_files += [(folder, f) for f in glob.glob(os.path.join(folder, f'*.{ext}'))]

            total = len(all_files)
            self._total_start = time.time()
            self._set_total(0, total)

            for idx, (folder, f) in enumerate(all_files):
                if self._stop_event.is_set():
                    print("\n已停止。")
                    break
                out_dir = os.path.join(folder, 'trans')
                os.makedirs(out_dir, exist_ok=True)
                args.output = out_dir

                name = os.path.basename(f)
                self._set_total(idx, total, name)
                self._file_start = time.time()
                self._set_file(0, 1)
                print(f"\n[{idx+1}/{total}] {f}")
                transcode.process_file(f, args, encoder,
                                       progress_cb=self._set_file,
                                       stop_event=self._stop_event)
                self._set_total(idx + 1, total, name)

            if not self._stop_event.is_set():
                print("\n全部完成。")
            self.after(0, lambda: self.file_label.configure(text='当前文件: -'))
            self.after(0, lambda: self.file_bar.configure(value=0))
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
