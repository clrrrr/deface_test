#!/usr/bin/env python3
import argparse
import subprocess
import os
import glob
import sys
import io
import time
import json
import shutil
import cv2
from tqdm import tqdm
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# Get ffprobe from vendor directory (relative to this script)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_ffprobe_name = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
FFPROBE = os.path.join(_script_dir, 'vendor', _ffprobe_name)

RESOLUTIONS = {
    '480p':  '854:480',
    '720p':  '1280:720',
    '1080p': '1920:1080',
    '4k':    '3840:2160',
}


def get_rotation(path):
    """Get video rotation in degrees using ffprobe. Normalizes to 0-360 range."""
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    try:
        r = subprocess.run(
            [FFPROBE, '-v', 'quiet', '-print_format', 'json', '-show_streams', path],
            capture_output=True, text=True, **kwargs
        )
        if r.returncode != 0:
            return 0
        data = json.loads(r.stdout)
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Try tags first (older format)
                rotation = stream.get('tags', {}).get('rotate')
                if rotation:
                    rot = int(rotation)
                    return rot % 360

                # Try side_data_list for Display Matrix rotation (newer format)
                for side_data in stream.get('side_data_list', []):
                    if side_data.get('side_data_type') == 'Display Matrix':
                        rot = side_data.get('rotation')
                        if rot is not None:
                            return int(rot) % 360
    except:
        return 0
    return 0


# (codec, gpu) -> ffmpeg encoder
ENCODERS = {
    ('hevc', 'nvidia'): 'hevc_nvenc',
    ('hevc', 'amd'):    'hevc_amf',
    ('hevc', 'intel'):  'hevc_qsv',
    ('hevc', 'cpu'):    'libx265',
    ('h264', 'nvidia'): 'h264_nvenc',
    ('h264', 'amd'):    'h264_amf',
    ('h264', 'intel'):  'h264_qsv',
    ('h264', 'cpu'):    'libx264',
    ('vp9',  'cpu'):    'libvpx-vp9',
    ('av1',  'cpu'):    'libaom-av1',
}


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    cap.release()

    rotation = get_rotation(path)
    # 90° or 270° rotation means width/height are swapped
    if abs(rotation) in (90, 270):
        w, h = h, w

    size = os.path.getsize(path) if os.path.exists(path) else 0
    duration = total / fps
    bitrate = int(size * 8 / duration / 1000) if duration > 0 else 0
    fmt = os.path.splitext(path)[1].lstrip('.')
    return {'fps': fps, 'nframes': total, 'width': w, 'height': h,
            'size': size, 'bitrate': bitrate, 'codec': codec, 'format': fmt, 'rotation': rotation}


def detect_gpu():
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}
    for gpu, enc in [('nvidia', 'hevc_nvenc'), ('amd', 'hevc_amf'), ('intel', 'hevc_qsv')]:
        r = subprocess.run(
            [FFMPEG, '-f', 'lavfi', '-i', 'nullsrc=s=64x64', '-t', '0.1',
             '-c:v', enc, '-f', 'null', '-'],
            capture_output=True, **kwargs
        )
        if r.returncode == 0:
            return gpu
    return 'cpu'


def build_cmd(input_path, output_path, args, encoder, info, target_br=None):
    cmd = [FFMPEG, '-y', '-hide_banner', '-hwaccel', 'auto']

    if args.start_frame > 0:
        cmd += ['-ss', str(args.start_frame / info['fps'])]

    cmd += ['-i', input_path]

    if args.end_frame >= 0:
        cmd += ['-frames:v', str(args.end_frame - args.start_frame)]

    filters = []
    # Apply rotation correction first to make video upright
    rotation = info.get('rotation', 0)
    if rotation == 90:
        filters.append('transpose=1')  # 90° clockwise
    elif rotation == 180:
        filters.append('transpose=2,transpose=2')  # 180°
    elif rotation == 270:
        filters.append('transpose=2')  # 90° counter-clockwise

    if args.fps is not None:
        filters.append(f"fps={args.fps}")
    if args.resolution != 'original':
        filters.append(f"scale={RESOLUTIONS[args.resolution]}")
    if filters:
        cmd += ['-vf', ','.join(filters)]

    br = target_br if target_br is not None else args.bitrate
    bufsize = br * 2
    cmd += ['-c:v', encoder, '-b:v', f'{br}k', '-threads', '0']
    cmd += ['-minrate', f'{br}k', '-maxrate', f'{br}k', '-bufsize', f'{bufsize}k']
    # 各编码器强制 CBR
    if encoder.endswith('_nvenc') or encoder.endswith('_amf'):
        cmd += ['-rc', 'cbr']
    elif encoder.endswith('_qsv'):
        # QSV: min==max==target 时自动进入 CBR 模式，无需额外参数
        pass
    elif encoder == 'libx264':
        cmd += ['-x264-params', 'nal-hrd=cbr']
    elif encoder == 'libx265':
        cmd += ['-x265-params',
                f'vbv-maxrate={br}:vbv-minrate={br}:vbv-bufsize={bufsize}:strict-cbr=1']
    elif encoder == 'libaom-av1':
        cmd += ['-aom-params', 'end-usage=cbr']
    if encoder in ('libx265', 'libx264', 'libvpx-vp9'):
        cmd += ['-preset', args.preset]
    if not args.keep_audio:
        cmd += ['-an']
    cmd += ['-progress', 'pipe:1', '-nostats']
    cmd += [output_path]
    return cmd


def run_with_progress(cmd, n_frames, label, progress_cb=None, stop_event=None):
    kwargs = {}
    if sys.platform == 'win32':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, **kwargs)
    tqdm_out = None if progress_cb is None else io.StringIO()
    with tqdm(total=n_frames, unit='frame', ncols=90, desc=label, file=tqdm_out) as pbar:
        current = 0
        for line in proc.stdout:
            if stop_event and stop_event.is_set():
                proc.terminate()
                break
            if line.startswith('frame='):
                try:
                    f = int(line.split('=')[1].strip())
                    pbar.update(f - current)
                    current = f
                    if progress_cb:
                        progress_cb(current, n_frames)
                except ValueError:
                    pass
        pbar.update(n_frames - current)
        if progress_cb:
            progress_cb(n_frames, n_frames)
    proc.wait()
    return proc.returncode


def process_file(input_path, args, encoder, progress_cb=None, stop_event=None):
    info = get_video_info(input_path)
    start = args.start_frame
    end = info['nframes'] if args.end_frame < 0 else min(args.end_frame, info['nframes'])
    n_frames = end - start

    dirname = os.path.dirname(os.path.abspath(input_path))
    basename = os.path.splitext(os.path.basename(input_path))[0]
    if args.output and not os.path.isdir(args.output):
        output_path = args.output
    else:
        out_dir = args.output if (args.output and os.path.isdir(args.output)) else dirname
        suffix = f"_{start}_{end}_comp" if (start != 0 or args.end_frame >= 0) else "_comp"
        output_path = os.path.join(out_dir, f"{basename}{suffix}.{args.fmt}")

    print(f"\n[Input Video Info]  {input_path}")
    print(f"  size:       {info['size'] / 1024 / 1024:.2f} MB")
    print(f"  nframes:    {info['nframes']}")
    print(f"  bitrate:    {info['bitrate']} kbps")
    print(f"  fps:        {info['fps']:.2f}")
    print(f"  resolution: {info['width']}x{info['height']}")
    rotation = info.get('rotation', 0)
    if rotation:
        print(f"  [DEBUG] 识别到rotation信息：{rotation}度")
    else:
        print(f"  [DEBUG] 未识别到rotation信息")
    print(f"  codec:      {info['codec']}")
    print(f"  format:     {info['format']}")

    out_res = RESOLUTIONS.get(args.resolution, f"{info['width']}:{info['height']}")
    print(f"\n[Processing Config]")
    print(f"  output:     {output_path}")
    print(f"  frames:     {start} -> {end}  ({n_frames} frames)")
    fps_str = str(args.fps) if args.fps else f"{info['fps']:.2f} (original)"
    print(f"  fps:        {fps_str}  bitrate: {args.bitrate} kbps")
    print(f"  resolution: {out_res.replace(':', 'x')}  codec: {args.codec} ({encoder})  preset: {args.preset}")
    print()

    user_br = args.bitrate
    duration = n_frames / info['fps'] if info['fps'] > 0 else 0

    def _encode(target_br, label_suffix=''):
        c = build_cmd(input_path, output_path, args, encoder, info, target_br=target_br)
        rc = run_with_progress(c, n_frames,
                               os.path.basename(input_path) + label_suffix,
                               progress_cb, stop_event)
        if rc != 0 or duration <= 0:
            return rc, 0
        b = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        return rc, int(b * 8 / duration / 1000)

    # 首轮：按设定值严格 CBR 编码
    rc, actual_br = _encode(user_br)

    # 若实测低于设定值，按比例提一档重编（仅一次，避免无限循环）
    if rc == 0 and duration > 0 and actual_br < user_br and not (stop_event and stop_event.is_set()):
        # 提档比例：补足缺口 + 10% 余量，下限 +15%，上限 +50%
        ratio = max(1.15, min(1.50, (user_br / max(actual_br, 1)) * 1.10))
        boosted = int(user_br * ratio)
        print(f"  [Retry] 实测 {actual_br} kbps < 设定 {user_br} kbps，提升目标至 {boosted} kbps 重编")
        rc, actual_br = _encode(boosted, ' (retry)')

    if rc == 0:
        out_bytes = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        out_mb = out_bytes / 1024 / 1024
        msg = f"Done -> {output_path}  ({out_mb:.2f} MB, {actual_br} kbps, 设定 {user_br} kbps)"
        if duration > 0 and actual_br < user_br:
            msg += f"  [WARN] 重编后仍低于设定，可能受片段长度/内容复杂度限制"
        print(msg)
    else:
        print(f"Error: ffmpeg exited with code {rc}")
    return rc


def main():
    parser = argparse.ArgumentParser(description='Fast video transcoder (GPU-accelerated)')
    parser.add_argument('input', nargs='+', help='Input file(s) or directory')
    parser.add_argument('-o', '--output', help='Output path (file or directory for batch)')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=-1)
    parser.add_argument('--fps', type=float, default=None, help='Output fps (default: keep original)')
    parser.add_argument('--bitrate', type=int, default=1000, help='kbps')
    parser.add_argument('--resolution', default='720p',
                        choices=list(RESOLUTIONS.keys()) + ['original'])
    parser.add_argument('--codec', default='hevc', choices=['hevc', 'h264', 'vp9', 'av1'])
    parser.add_argument('--preset', default='fast',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'])
    parser.add_argument('--gpu', default='auto',
                        choices=['auto', 'nvidia', 'amd', 'intel', 'cpu'])
    parser.add_argument('--format', default='mp4', dest='fmt')
    parser.add_argument('--keep-audio', action='store_true', help='Keep audio (default: mute)')
    args = parser.parse_args()

    # Resolve GPU
    gpu = detect_gpu() if args.gpu == 'auto' else args.gpu
    encoder = ENCODERS.get((args.codec, gpu)) or ENCODERS.get((args.codec, 'cpu'))
    print(f"GPU: {gpu}  Encoder: {encoder}")

    # Collect input files
    files = []
    for pattern in args.input:
        if os.path.isdir(pattern):
            for ext in ('mp4', 'MP4', 'mov', 'MOV', 'avi', 'AVI', 'mkv'):
                files += glob.glob(os.path.join(pattern, f'*.{ext}'))
        else:
            files += glob.glob(pattern) or ([pattern] if os.path.isfile(pattern) else [])

    if not files:
        print("No input files found.")
        return 1

    t0 = time.time()
    errors = 0
    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {f}")
        rc = process_file(f, args, encoder)
        if rc != 0:
            errors += 1

    elapsed = time.time() - t0
    print(f"\nFinished {len(files)} file(s) in {elapsed:.1f}s  ({errors} error(s))")


if __name__ == '__main__':
    main()
