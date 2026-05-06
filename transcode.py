#!/usr/bin/env python3
import argparse
import subprocess
import os
import glob
import time
import cv2
from tqdm import tqdm
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

RESOLUTIONS = {
    '480p':  '854:480',
    '720p':  '1280:720',
    '1080p': '1920:1080',
    '4k':    '3840:2160',
}

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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    cap.release()
    size = os.path.getsize(path)
    duration = total / fps if fps > 0 else 0
    bitrate = int(size * 8 / duration / 1000) if duration > 0 else 0
    fmt = os.path.splitext(path)[1].lstrip('.')
    return {'fps': fps, 'nframes': total, 'width': w, 'height': h,
            'size': size, 'bitrate': bitrate, 'codec': codec, 'format': fmt}


def detect_gpu():
    for gpu, enc in [('nvidia', 'hevc_nvenc'), ('amd', 'hevc_amf'), ('intel', 'hevc_qsv')]:
        r = subprocess.run(
            [FFMPEG, '-f', 'lavfi', '-i', 'nullsrc=s=64x64', '-t', '0.1',
             '-c:v', enc, '-f', 'null', '-'],
            capture_output=True
        )
        if r.returncode == 0:
            return gpu
    return 'cpu'


def build_cmd(input_path, output_path, args, encoder, info):
    cmd = [FFMPEG, '-y', '-hide_banner', '-hwaccel', 'auto']

    if args.start_frame > 0:
        cmd += ['-ss', str(args.start_frame / info['fps'])]

    cmd += ['-i', input_path]

    if args.end_frame >= 0:
        cmd += ['-frames:v', str(args.end_frame - args.start_frame)]

    filters = []
    if args.fps is not None:
        filters.append(f"fps={args.fps}")
    if args.resolution != 'original':
        filters.append(f"scale={RESOLUTIONS[args.resolution]}")
    if filters:
        cmd += ['-vf', ','.join(filters)]

    cmd += ['-c:v', encoder, '-b:v', f'{args.bitrate}k', '-threads', '0']
    if encoder in ('libx265', 'libx264', 'libvpx-vp9'):
        cmd += ['-preset', args.preset]
    cmd += ['-progress', 'pipe:1', '-nostats']
    cmd += [output_path]
    return cmd


def run_with_progress(cmd, n_frames, label):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    with tqdm(total=n_frames, unit='frame', ncols=90, desc=label) as pbar:
        current = 0
        for line in proc.stdout:
            if line.startswith('frame='):
                try:
                    f = int(line.split('=')[1].strip())
                    pbar.update(f - current)
                    current = f
                except ValueError:
                    pass
        pbar.update(n_frames - current)
    proc.wait()
    return proc.returncode


def process_file(input_path, args, encoder):
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
        suffix = f"_{start}_{end}_compress" if (start != 0 or args.end_frame >= 0) else "_full_compress"
        output_path = os.path.join(out_dir, f"{basename}{suffix}.{args.fmt}")

    print(f"\n[Input Video Info]  {input_path}")
    print(f"  size:       {info['size'] / 1024 / 1024:.2f} MB")
    print(f"  nframes:    {info['nframes']}")
    print(f"  bitrate:    {info['bitrate']} kbps")
    print(f"  fps:        {info['fps']:.2f}")
    print(f"  resolution: {info['width']}x{info['height']}")
    print(f"  codec:      {info['codec']}")
    print(f"  format:     {info['format']}")

    out_res = RESOLUTIONS.get(args.resolution, f"{info['width']}:{info['height']}")
    print(f"\n[Processing Config]")
    print(f"  output:     {output_path}")
    print(f"  frames:     {start} -> {end}  ({n_frames} frames)")
    print(f"  fps:        {args.fps if args.fps else f'{info[\"fps\"]:.2f} (original)'}  bitrate: {args.bitrate} kbps")
    print(f"  resolution: {out_res.replace(':', 'x')}  codec: {args.codec} ({encoder})  preset: {args.preset}")
    print()

    cmd = build_cmd(input_path, output_path, args, encoder, info)
    rc = run_with_progress(cmd, n_frames, os.path.basename(input_path))

    if rc == 0:
        out_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"Done -> {output_path}  ({out_size:.2f} MB)")
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
