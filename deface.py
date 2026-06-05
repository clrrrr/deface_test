#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import mimetypes
import os
import queue
import re
import subprocess
import threading
import time
from typing import Dict, Tuple

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import imageio.plugins.ffmpeg
import cv2
import imageio_ffmpeg

from centerface import CenterFace

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

__version__ = '1.5.0-local'

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _find_ffprobe():
    """Locate ffprobe binary. Try alongside imageio_ffmpeg's ffmpeg, then PATH."""
    import shutil
    cand = os.path.join(os.path.dirname(FFMPEG), 'ffprobe')
    for c in (cand, cand + '.exe'):
        if os.path.isfile(c):
            return c
    return shutil.which('ffprobe')


FFPROBE = _find_ffprobe()

# codec_name from ffprobe -> ffmpeg encoder (CPU)
CODEC_TO_ENCODER = {
    'h264':       'libx264',
    'hevc':       'libx265',
    'h265':       'libx265',
    'vp9':        'libvpx-vp9',
    'vp8':        'libvpx',
    'av1':        'libaom-av1',
    'mpeg4':      'mpeg4',
    'mpeg2video': 'mpeg2video',
}

VALID_ENCODERS = [
    'libx264', 'libx265', 'libvpx-vp9', 'libvpx', 'libaom-av1',
    'h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'h264_qsv', 'hevc_qsv'
]


class Profile:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.times = {}
        self.resources = []
        self.queue_samples = []
        self._stop = threading.Event()
        self._thread = None

    def start_sampling(self, raw_q, result_q):
        if not self.enabled or not HAS_PSUTIL:
            return
        def _sample():
            while not self._stop.is_set():
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                gpu_util = gpu_mem = 0
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(0)
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    gpu_util = u.gpu
                    gpu_mem = m.used / m.total * 100
                except:
                    pass
                self.resources.append((time.time(), cpu, ram, gpu_util, gpu_mem))
                self.queue_samples.append((raw_q.qsize(), result_q.qsize()))
                time.sleep(2)
        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()

    def stop_sampling(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=1)

    def timer(self, name):
        class _Timer:
            def __init__(self, prof, n):
                self.prof = prof
                self.name = n
            def __enter__(self):
                self.t0 = time.time()
                return self
            def __exit__(self, *args):
                if self.prof.enabled:
                    self.prof.times[self.name] = time.time() - self.t0
        return _Timer(self, name)

    def report(self, raw_maxsize, result_maxsize):
        if not self.enabled:
            return
        print('\n=== Profile Report ===')
        for k, v in self.times.items():
            print(f'  {k}: {v:.2f}s')
        if self.resources:
            cpu = np.mean([r[1] for r in self.resources])
            ram = np.mean([r[2] for r in self.resources])
            gpu = np.mean([r[3] for r in self.resources])
            gmem = np.mean([r[4] for r in self.resources])
            print(f'  avg CPU: {cpu:.1f}%  RAM: {ram:.1f}%  GPU: {gpu:.1f}%  GMEM: {gmem:.1f}%')
        if self.queue_samples:
            raw_full = sum(1 for r, _ in self.queue_samples if r >= raw_maxsize) / len(self.queue_samples) * 100
            result_full = sum(1 for _, res in self.queue_samples if res >= result_maxsize) / len(self.queue_samples) * 100
            result_empty = sum(1 for _, res in self.queue_samples if res == 0) / len(self.queue_samples) * 100
            print(f'  raw_queue full: {raw_full:.1f}%  result_queue full: {result_full:.1f}%  empty: {result_empty:.1f}%')
            if result_full > 60:
                print('  [Diagnosis] Encoder bottleneck (result_queue mostly full)')
            elif result_empty > 60 and gpu < 50:
                print('  [Diagnosis] Inference bottleneck (result_queue empty, low GPU)')
            elif result_empty > 60 and raw_full < 20:
                print('  [Diagnosis] I/O bottleneck (both queues empty)')


def _clamp_8bit_pix_fmt(pf):
    """Drop bit-depth suffix from pix_fmt since we encode from rgb24 (8-bit)."""
    if not pf:
        return 'yuv420p'
    for suf in ('10le', '10be', '12le', '12be', '14le', '14be', '16le', '16be'):
        if pf.endswith(suf):
            return pf[:-len(suf)]
    return pf


def probe_video(path):
    """Probe video parameters via ffprobe (preferred) or ffmpeg stderr fallback.
    Returns dict with keys: width, height, fps, fps_str, nframes, duration,
    codec, pix_fmt, bitrate_k, color_space, color_primaries, color_transfer, color_range.
    """
    info = {'width': 0, 'height': 0, 'fps': 0.0, 'fps_str': '0',
            'nframes': 0, 'duration': 0.0, 'codec': 'h264',
            'pix_fmt': 'yuv420p', 'bitrate_k': 0,
            'color_space': None, 'color_primaries': None,
            'color_transfer': None, 'color_range': None}
    size_bytes = os.path.getsize(path) if os.path.isfile(path) else 0

    if FFPROBE:
        try:
            r = subprocess.run(
                [FFPROBE, '-v', 'error', '-print_format', 'json',
                 '-show_format', '-show_streams', '-select_streams', 'v:0', path],
                capture_output=True, text=True, timeout=30)
            data = json.loads(r.stdout)
            streams = data.get('streams', [])
            if not streams:
                return None
            s = streams[0]
            fmt = data.get('format', {})
            info['width']  = int(s.get('width') or 0)
            info['height'] = int(s.get('height') or 0)
            info['codec']  = s.get('codec_name') or info['codec']
            info['pix_fmt'] = s.get('pix_fmt') or info['pix_fmt']
            info['color_space']     = s.get('color_space') or None
            info['color_primaries'] = s.get('color_primaries') or None
            info['color_transfer']  = s.get('color_transfer') or None
            info['color_range']     = s.get('color_range') or None

            fps_str = s.get('r_frame_rate') or s.get('avg_frame_rate') or '0/1'
            info['fps_str'] = fps_str
            try:
                num, den = fps_str.split('/')
                info['fps'] = float(num) / float(den) if float(den) else 0.0
            except Exception:
                info['fps'] = float(fps_str or 0)

            info['duration'] = float(s.get('duration') or fmt.get('duration') or 0)
            nb = s.get('nb_frames') or fmt.get('nb_frames')
            if nb:
                info['nframes'] = int(nb)
            elif info['fps'] > 0 and info['duration'] > 0:
                info['nframes'] = int(round(info['fps'] * info['duration']))

            # Video bitrate: stream first, else format, else filesize estimate
            br = s.get('bit_rate') or fmt.get('bit_rate')
            if br:
                info['bitrate_k'] = int(int(br) / 1000)
            elif info['duration'] > 0 and size_bytes > 0:
                info['bitrate_k'] = int(size_bytes * 8 / info['duration'] / 1000)
            return info
        except Exception as e:
            print(f'  [probe] ffprobe failed: {e}, falling back to cv2')

    # Fallback: cv2 + filesize (loses color metadata)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info['width']    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info['height']   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info['fps']      = cap.get(cv2.CAP_PROP_FPS) or 0.0
    info['fps_str']  = f'{info["fps"]}'
    info['nframes']  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info['duration'] = info['nframes'] / info['fps'] if info['fps'] > 0 else 0.0
    cap.release()
    if info['duration'] > 0 and size_bytes > 0:
        info['bitrate_k'] = int(size_bytes * 8 / info['duration'] / 1000)
    return info


def build_encoder_args(encoder, probe, target_k):
    """Build ffmpeg encoder argument list (no input/output, no -an)."""
    pix_fmt = _clamp_8bit_pix_fmt(probe.get('pix_fmt'))
    bufk = max(target_k * 2, 1)
    args = [
        '-c:v', encoder,
        '-pix_fmt', pix_fmt,
        '-b:v', f'{target_k}k',
        '-minrate', f'{target_k}k',
        '-maxrate', f'{target_k}k',
        '-bufsize', f'{bufk}k',
        '-threads', '0',
    ]
    if encoder in ('h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf'):
        args += ['-rc', 'cbr']
    elif encoder == 'libx264':
        args += ['-x264-params', 'nal-hrd=cbr']
    elif encoder == 'libx265':
        args += ['-x265-params',
                 f'vbv-maxrate={target_k}:vbv-minrate={target_k}:vbv-bufsize={bufk}:strict-cbr=1']
    elif encoder == 'libaom-av1':
        args += ['-aom-params', 'end-usage=cbr']
    # Color metadata passthrough
    for src_key, dst_flag in (('color_space', '-colorspace'),
                              ('color_primaries', '-color_primaries'),
                              ('color_transfer', '-color_trc'),
                              ('color_range', '-color_range')):
        v = probe.get(src_key)
        if v and v not in ('unknown', 'reserved', 'N/A'):
            args += [dst_flag, v]
    return args


def build_writer_cmd_file(opath, w, h, fps_str, encoder, probe, target_k, preset=None, src_path=None):
    """ffmpeg cmd: raw rgb24 stdin -> encoded file (passthrough source params + metadata)."""
    cmd = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', fps_str,
           '-i', 'pipe:0']
    if src_path:
        # Second input: source file. Used only for container metadata.
        cmd += ['-i', src_path, '-map', '0:v', '-map_metadata', '1']
    cmd += ['-an']
    cmd += build_encoder_args(encoder, probe, target_k)
    if preset and encoder in ('libx264', 'libx265', 'libvpx-vp9'):
        cmd += ['-preset', preset]
    cmd += [opath]
    return cmd


def build_writer_cmd_cam(opath, w, h, fps, encoder='libx264', preset=None):
    """Cam fallback: no source to passthrough, use libx264/CRF defaults."""
    cmd = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps),
           '-i', 'pipe:0', '-an',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18']
    if preset:
        cmd += ['-preset', preset]
    cmd += [opath]
    return cmd


def measure_bitrate_k(path):
    """Return file's video bitrate in kbps (via ffprobe if available, else filesize/duration)."""
    if FFPROBE:
        try:
            r = subprocess.run(
                [FFPROBE, '-v', 'error', '-print_format', 'json',
                 '-show_format', '-show_streams', '-select_streams', 'v:0', path],
                capture_output=True, text=True, timeout=30)
            data = json.loads(r.stdout)
            s = (data.get('streams') or [{}])[0]
            fmt = data.get('format', {})
            br = s.get('bit_rate') or fmt.get('bit_rate')
            if br:
                return int(int(br) / 1000)
            dur = float(s.get('duration') or fmt.get('duration') or 0)
            sz = os.path.getsize(path) if os.path.isfile(path) else 0
            if dur > 0 and sz > 0:
                return int(sz * 8 / dur / 1000)
        except Exception:
            pass
    # Fallback
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = nframes / fps if fps > 0 else 0
    sz = os.path.getsize(path) if os.path.isfile(path) else 0
    return int(sz * 8 / dur / 1000) if dur > 0 and sz > 0 else 0


def re_encode_bitrate(opath, encoder, probe, target_k, preset=None, src_path=None):
    """Re-encode opath in-place with higher bitrate target. Used when first pass undershoots.
    If src_path is given, container metadata is mapped from there (not from the intermediate)."""
    tmp = opath + '.tmp_rebr' + os.path.splitext(opath)[1]
    cmd = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error', '-i', opath]
    if src_path:
        cmd += ['-i', src_path, '-map', '0:v', '-map_metadata', '1']
    cmd += ['-an']
    cmd += build_encoder_args(encoder, probe, target_k)
    if preset and encoder in ('libx264', 'libx265', 'libvpx-vp9'):
        cmd += ['-preset', preset]
    cmd += [tmp]
    rc = subprocess.run(cmd, capture_output=True).returncode
    if rc == 0:
        os.replace(tmp, opath)
    elif os.path.exists(tmp):
        os.remove(tmp)
    return rc


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        pass
    if draw_scores:
        cv2.putText(
            frame, f'{score:.2f}', (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0)
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def video_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        replaceimg = None,
        mosaicsize: int = 20,
        batchsize: int = 8,
        prefetch: int = 2,
        preset: str = None,
        bitrate_margin: float = 1.30,
        profile: Profile = None,
        encoder: str = 'auto',
):
    if profile is None:
        profile = Profile(enabled=False)

    cam_reader = None
    probe = None
    if cam:
        try:
            cam_reader = imageio.get_reader(ipath)
            meta = cam_reader.get_meta_data()
            fps = meta['fps']
            w, h = meta['size']
            nframes = None
            fps_str = str(fps)
        except:
            print(f'Could not find video device {ipath}. Please set a valid input.')
            return
    else:
        with profile.timer('probe'):
            probe = probe_video(ipath)
        if probe is None or probe['width'] == 0:
            print(f'Could not probe {ipath} as a video file. Skipping file...')
            return
        w, h = probe['width'], probe['height']
        fps = probe['fps']
        fps_str = probe['fps_str']
        nframes = probe['nframes']
        print(f'  [probe] {probe["codec"]} {w}x{h} {fps:.3f}fps {probe["pix_fmt"]} '
              f'{probe["bitrate_k"]}kbps  color=({probe["color_space"]}/{probe["color_primaries"]}/'
              f'{probe["color_transfer"]}/{probe["color_range"]})')

    if nested:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1, leave=True)
    else:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes)

    writer_proc = None
    target_k = 0
    chosen_encoder = encoder
    if opath is not None:
        if probe is not None:
            if encoder == 'auto':
                chosen_encoder = CODEC_TO_ENCODER.get(probe['codec'], 'libx264')
            target_k = max(int(probe['bitrate_k'] * bitrate_margin), 1)
            cmd = build_writer_cmd_file(opath, w, h, fps_str, chosen_encoder, probe, target_k,
                                        preset=preset, src_path=ipath)
            print(f'  [encode] encoder={chosen_encoder} target={target_k}k preset={preset}')
        else:
            if encoder == 'auto':
                chosen_encoder = 'libx264'
            cmd = build_writer_cmd_cam(opath, w, h, fps, encoder=chosen_encoder, preset=preset)
            print(f'  [encode] encoder={chosen_encoder} preset={preset}')
        writer_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    BATCH_SIZE = batchsize
    total_frames = 0
    face_frames = 0
    raw_queue = queue.Queue(maxsize=prefetch)
    result_queue = queue.Queue(maxsize=prefetch)

    profile.start_sampling(raw_queue, result_queue)

    def _reader():
        if cam:
            buf = []
            for frame in cam_read_iter(cam_reader):
                buf.append(frame)
                if len(buf) >= BATCH_SIZE:
                    raw_queue.put(buf)
                    buf = []
            if buf:
                raw_queue.put(buf)
        else:
            cmd = [FFMPEG, '-hwaccel', 'auto', '-threads', '0', '-i', ipath,
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            frame_bytes = w * h * 3
            buf = []
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
                buf.append(frame)
                if len(buf) >= BATCH_SIZE:
                    raw_queue.put(buf)
                    buf = []
            if buf:
                raw_queue.put(buf)
            proc.wait()
        raw_queue.put(None)

    def _writer():
        while True:
            item = result_queue.get()
            if item is None:
                break
            for f, dets in item:
                anonymize_frame(dets, f, mask_scale=mask_scale,
                    replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
                    replaceimg=replaceimg, mosaicsize=mosaicsize)
                if writer_proc is not None:
                    try:
                        writer_proc.stdin.write(f.tobytes())
                    except BrokenPipeError:
                        return
                if enable_preview:
                    cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', f[:, :, ::-1])
                    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                        cv2.destroyAllWindows()
                        return
                bar.update()

    threading.Thread(target=_reader, daemon=True).start()
    writer_t = threading.Thread(target=_writer, daemon=True)
    writer_t.start()

    with profile.timer('encode_wall'):
        while True:
            buf = raw_queue.get()
            if buf is None:
                result_queue.put(None)
                break
            batch_results = centerface.batch_call(buf, threshold=threshold)
            pairs = []
            for f, (dets, _) in zip(buf, batch_results):
                total_frames += 1
                if len(dets) > 0:
                    face_frames += 1
                pairs.append((f, dets))
            result_queue.put(pairs)

        writer_t.join()
        if cam_reader is not None:
            cam_reader.close()
        if writer_proc is not None:
            try:
                writer_proc.stdin.close()
            except Exception:
                pass
            writer_proc.wait()
        bar.close()

    # Strict bitrate guarantee: re-encode if first pass undershot the source
    if probe is not None and opath is not None and writer_proc is not None \
            and writer_proc.returncode == 0 and probe['bitrate_k'] > 0:
        actual_k = measure_bitrate_k(opath)
        src_k = probe['bitrate_k']
        if 0 < actual_k < src_k:
            with profile.timer('retry'):
                ratio = src_k / actual_k
                new_target = max(int(target_k * ratio * 1.20), int(src_k * 1.50))
                print(f'  [bitrate retry] actual {actual_k}k < src {src_k}k, '
                      f're-encoding @ {new_target}k')
                rc = re_encode_bitrate(opath, chosen_encoder, probe, new_target, preset=preset, src_path=ipath)
                if rc != 0:
                    print(f'  [bitrate retry] re-encode failed (rc={rc})')
                else:
                    final_k = measure_bitrate_k(opath)
                    if final_k < src_k:
                        print(f'  [WARN] post-retry bitrate {final_k}k still < src {src_k}k')

    profile.stop_sampling()
    profile.report(prefetch, prefetch)

    return total_frames, face_frames


def image_detect(
        ipath: str,
        opath: str,
        centerface: CenterFace,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
):
    frame = iio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        # Save image with EXIF metadata
        imageio.imsave(opath, frame, exif=exif_dict)

    # print(f'Output saved to {opath}')


def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg
    )

    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ deface vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ deface cam` (which is a shortcut for `$ deface -p \'<video0>\'`.')
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path + postfix "_anonymized".')
    parser.add_argument(
        '--thresh', '-t', default=0.2, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.2.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=False, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--bitrate-margin', default=1.30, type=float, metavar='M',
        help='Output bitrate target = source_bitrate * M. Output is guaranteed >= source via CBR + retry. Default: 1.30.')
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--version', action='version', version=__version__,
        help='Print version number and exit.')
    parser.add_argument(
        '--keep-metadata', '-m', default=False, action='store_true',
        help='Keep metadata of the original image. Default : False.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    # Performance options
    parser.add_argument('--preset', default='fast',
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'],
        help='Encoder preset for output video (default: fast)')
    parser.add_argument('--batchsize', type=int, default=8, metavar='N',
        help='Batch size for face detection inference (default: 8)')
    parser.add_argument('--prefetch', type=int, default=2, metavar='N',
        help='Queue depth for frame prefetch (default: 2)')
    parser.add_argument('--profile', default=False, action='store_true',
        help='Enable performance profiling (timing, resource monitoring, bottleneck diagnosis)')
    parser.add_argument('--encoder', default='auto',
        choices=['auto'] + VALID_ENCODERS,
        help='Video encoder (default: auto - match source codec). Use libx264 for speed, GPU encoders (h264_nvenc, hevc_nvenc) if available.')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)

    
    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata

    replaceimg = None
    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    centerface = CenterFace(in_shape=in_shape, backend=backend, override_execution_provider=execution_provider)

    prof = Profile(enabled=args.profile)

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anonymized{ext}'
        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')
        if filetype == 'video' or is_cam:
            result = video_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                cam=is_cam,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                batchsize=args.batchsize,
                prefetch=args.prefetch,
                preset=args.preset,
                bitrate_margin=args.bitrate_margin,
                profile=prof,
                encoder=args.encoder,
            )
            if result is not None and not is_cam:
                total_frames, face_frames = result
                ratio = face_frames / total_frames if total_frames > 0 else 0.0
                csv_path = os.path.join(os.path.dirname(os.path.abspath(ipath)), 'face_stats.csv')
                file_exists = os.path.isfile(csv_path)
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvf:
                    writer_csv = csv.writer(csvf)
                    if not file_exists:
                        writer_csv.writerow(['filename', 'total_frames', 'face_frames', 'face_ratio'])
                    writer_csv.writerow([os.path.basename(ipath), total_frames, face_frames, f'{ratio:.4f}'])
                print(f'Stats: {face_frames}/{total_frames} frames with faces ({ratio:.1%}) -> {csv_path}')
        elif filetype == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=centerface,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
        elif filetype is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()
