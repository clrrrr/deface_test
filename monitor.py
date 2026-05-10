#!/usr/bin/env python3
import subprocess
import time
import psutil
import os

INTERVAL = 1.0


def get_gpu():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(', ')
            return {
                'gpu_util': int(parts[0]),
                'mem_util': int(parts[1]),
                'mem_used': int(parts[2]),
                'mem_total': int(parts[3]),
            }
    except Exception:
        pass
    return None


def get_ffmpeg_cpu():
    """Sum CPU% of all ffmpeg processes."""
    total = 0.0
    for p in psutil.process_iter(['name', 'cpu_percent']):
        try:
            if 'ffmpeg' in p.info['name'].lower():
                total += p.info['cpu_percent']
        except Exception:
            pass
    return total


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


# Prime cpu_percent (first call always returns 0)
psutil.cpu_percent(percpu=False)
for p in psutil.process_iter(['name', 'cpu_percent']):
    try:
        p.cpu_percent()
    except Exception:
        pass

print("Monitoring... Ctrl+C to stop\n")
time.sleep(INTERVAL)

while True:
    cpu_total = psutil.cpu_percent(percpu=False)
    mem = psutil.virtual_memory()
    ffmpeg_cpu = get_ffmpeg_cpu()
    gpu = get_gpu()

    clear()
    print(f"{'='*40}")
    print(f"  CPU total   : {cpu_total:5.1f}%")
    print(f"  ffmpeg procs: {ffmpeg_cpu:5.1f}%  (decode+encode)")
    print(f"  RAM used    : {mem.used/1024**3:.1f} / {mem.total/1024**3:.1f} GB  ({mem.percent:.0f}%)")
    if gpu:
        print(f"  GPU util    : {gpu['gpu_util']:3d}%")
        print(f"  GPU mem util: {gpu['mem_util']:3d}%")
        print(f"  GPU mem     : {gpu['mem_used']:5d} / {gpu['mem_total']} MB")
        bar_g = '#' * (gpu['gpu_util'] // 5) + '.' * (20 - gpu['gpu_util'] // 5)
        bar_c = '#' * (int(cpu_total) // 5) + '.' * (20 - int(cpu_total) // 5)
        print(f"\n  GPU [{bar_g}] {gpu['gpu_util']}%")
        print(f"  CPU [{bar_c}] {cpu_total:.0f}%")
        print()
        if gpu['gpu_util'] < 30 and cpu_total > 60:
            print("  >> BOTTLENECK: CPU (decode/anonymize)")
        elif gpu['gpu_util'] < 30 and ffmpeg_cpu > 40:
            print("  >> BOTTLENECK: ffmpeg decode")
        elif gpu['gpu_util'] > 80:
            print("  >> GPU saturated - try smaller batchsize or --scale")
        else:
            print("  >> balanced")
    else:
        print("  GPU: nvidia-smi not found")
    print(f"{'='*40}")

    time.sleep(INTERVAL)
