#!/usr/bin/env python3
import argparse
import time
import os
import cv2
import imageio
from tqdm import tqdm

RESOLUTIONS = {
    '480p': (854, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4k': (3840, 2160),
}
CODECS = {
    'hevc': 'libx265',
    'h264': 'libx264',
    'vp9': 'libvpx-vp9',
    'av1': 'libaom-av1',
}


def main():
    parser = argparse.ArgumentParser(description='Video transcoder')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int, default=-1)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--bitrate', type=int, default=1000, help='kbps')
    parser.add_argument('--resolution', default='720p',
                        choices=list(RESOLUTIONS.keys()) + ['original'])
    parser.add_argument('--codec', default='hevc', choices=list(CODECS.keys()))
    parser.add_argument('--format', default='mp4', dest='fmt')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open '{args.input}'")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    file_size = os.path.getsize(args.input)
    duration = total_frames / src_fps if src_fps > 0 else 0
    src_bitrate = file_size * 8 / duration / 1000 if duration > 0 else 0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    src_codec = ''.join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()
    src_fmt = os.path.splitext(args.input)[1].lstrip('.')

    start = args.start_frame
    end = total_frames if args.end_frame < 0 else min(args.end_frame, total_frames)
    n_frames = end - start

    if args.resolution == 'original':
        out_w, out_h = src_w, src_h
    else:
        out_w, out_h = RESOLUTIONS[args.resolution]

    if args.output is None:
        dirname = os.path.dirname(os.path.abspath(args.input))
        basename = os.path.splitext(os.path.basename(args.input))[0]
        if args.start_frame != 0 or args.end_frame >= 0:
            suffix = f"_{args.start_frame}_{end}_compress"
        else:
            suffix = "_full_compress"
        args.output = os.path.join(dirname, f"{basename}{suffix}.{args.fmt}")

    # Print source info (from metadata, no frame reading needed)
    print("\n[Input Video Info]")
    print(f"  size:       {file_size / 1024 / 1024:.2f} MB")
    print(f"  nframes:    {total_frames}")
    print(f"  bitrate:    {src_bitrate:.0f} kbps")
    print(f"  fps:        {src_fps:.2f}")
    print(f"  resolution: {src_w}x{src_h}")
    print(f"  codec:      {src_codec}")
    print(f"  format:     {src_fmt}")

    print("\n[Processing Config]")
    print(f"  input:      {args.input}")
    print(f"  output:     {args.output}")
    print(f"  frames:     {start} -> {end}  ({n_frames} frames)")
    print(f"  fps:        {args.fps}")
    print(f"  bitrate:    {args.bitrate} kbps")
    print(f"  resolution: {out_w}x{out_h}")
    print(f"  codec:      {args.codec} ({CODECS[args.codec]})")
    print(f"  format:     {args.fmt}")
    print()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec=CODECS[args.codec],
        bitrate=f'{args.bitrate}k',
        output_params=['-vf', f'scale={out_w}:{out_h}'],
    )

    t0 = time.time()
    with tqdm(total=n_frames, unit='frame', ncols=90) as pbar:
        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_frames - i - 1) if i > 0 else 0
            pbar.set_postfix(ETA=f'{eta:.0f}s')
            pbar.update(1)

    writer.close()
    cap.release()

    out_size = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone. Output: {args.output}  ({out_size:.2f} MB)")


if __name__ == '__main__':
    main()
