"""Profile per-stage latency, throughput and memory of the CENTINAL pipeline.

    python eval/eval_runtime.py --frames 100 --plot
    python eval/eval_runtime.py --frames 100 --device cpu   # compare backends
"""

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from collections import OrderedDict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centinal.models import BASE_DIR, load_csrnet, load_risk_head  # noqa: E402
from centinal.pipeline import (  # noqa: E402
    FeatureExtractor,
    RiskEngine,
    classify_scene,
    compute_risk,
    preprocess,
)
from centinal.viz import normalize_heatmap, overlay_heatmap  # noqa: E402


def synchronize(device):
    """Block until queued GPU work finishes, so timings are not understated."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize(values):
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile the CENTINAL pipeline")
    parser.add_argument("--video", type=str, default="videos/crowd_test.mp4")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3,
                        help="Frames to run before timing starts (lazy init, shader compile)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"], help="Override device selection")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output_dir", type=str, default="eval/results")
    args = parser.parse_args()

    video_path = args.video if os.path.isabs(args.video) else os.path.join(BASE_DIR, args.video)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else None
    csrnet, device, meta = load_csrnet(device=device)
    lstm, scaler, seq_len = load_risk_head()
    mode = meta["preprocess"]

    print(f"System : {platform.platform()}")
    print(f"Device : {device} | input: {mode} | LSTM window: {seq_len}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video  : {os.path.basename(video_path)} {width}x{height}")

    extractor = FeatureExtractor()
    engine = RiskEngine(lstm, scaler, seq_len)

    stages = OrderedDict((name, []) for name in
                         ["Frame Prep", "CSRNet Inference", "Feature Extraction",
                          "LSTM Inference", "Heatmap Generation", "Total Pipeline"])

    tracemalloc.start()
    processed = 0
    total_target = args.frames + args.warmup

    while processed < total_target:
        ok, frame = cap.read()
        if not ok:
            break
        timing = processed >= args.warmup
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess(rgb, device, mode)
        synchronize(device)
        t_prep = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with torch.no_grad():
            density_map = csrnet(tensor)
        synchronize(device)
        density = density_map.squeeze().cpu().numpy()
        t_csrnet = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        feats = extractor.extract(density)
        scene = classify_scene(feats.count)
        engine.push(feats)
        t_features = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        pred_idx, confidence = engine.predict()
        compute_risk(scene, pred_idx, confidence, engine.ready)
        t_lstm = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        heatmap, opacity = normalize_heatmap(density, scene, frame.shape[:2])
        overlay_heatmap(frame, heatmap, opacity)
        t_heatmap = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_start) * 1000

        if timing:
            stages["Frame Prep"].append(t_prep)
            stages["CSRNet Inference"].append(t_csrnet)
            stages["Feature Extraction"].append(t_features)
            stages["LSTM Inference"].append(t_lstm)
            stages["Heatmap Generation"].append(t_heatmap)
            stages["Total Pipeline"].append(t_total)

        processed += 1
        if processed % 25 == 0:
            print(f"  {processed}/{total_target} frames", flush=True)

    cap.release()
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not stages["Total Pipeline"]:
        print("No frames were timed.")
        return 1

    totals = np.array(stages["Total Pipeline"])
    fps = 1000.0 / totals

    print("\n" + "=" * 72)
    print("  RUNTIME PROFILE")
    print("=" * 72)
    print(f"  {'Stage':<22}{'mean ms':>10}{'p50':>9}{'p95':>9}{'max':>10}{'% total':>9}")
    print("  " + "-" * 68)
    mean_total = totals.mean()
    for name, values in stages.items():
        stats = summarize(values)
        share = "" if name == "Total Pipeline" else f"{stats['mean']/mean_total*100:8.1f}%"
        print(f"  {name:<22}{stats['mean']:10.2f}{stats['p50']:9.2f}"
              f"{stats['p95']:9.2f}{stats['max']:10.2f}{share:>9}")
    print("  " + "-" * 68)
    print(f"  Throughput : {fps.mean():.2f} FPS mean, {fps.min():.2f} min, {fps.max():.2f} max")
    print(f"  Peak Python heap: {peak_ram / 1024 / 1024:.1f} MB")
    if device.type == "cuda":
        print(f"  Peak VRAM  : {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB")
    print("=" * 72)

    profile = {
        "metadata": {
            "platform": platform.platform(),
            "device": str(device),
            "preprocess": mode,
            "seq_len": seq_len,
            "video": os.path.basename(video_path),
            "resolution": f"{width}x{height}",
            "frames_timed": len(totals),
            "warmup_frames": args.warmup,
            "average_fps": float(fps.mean()),
            "peak_python_heap_mb": peak_ram / 1024 / 1024,
        },
        "stages": {name: summarize(values) for name, values in stages.items()},
    }
    json_path = os.path.join(output_dir, f"runtime_profile_{device.type}.json")
    with open(json_path, "w") as fh:
        json.dump(profile, fh, indent=2)
    print(f"  Saved: {json_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [n for n in stages if n != "Total Pipeline"]
        means = [np.mean(stages[n]) for n in names]
        plt.figure(figsize=(9, 5))
        bars = plt.bar(names, means, color="#3b82f6", edgecolor="black", linewidth=0.5)
        for bar, value in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}",
                     ha="center", va="bottom", fontsize=9)
        plt.ylabel("Mean latency (ms)")
        plt.title(f"CENTINAL stage latency - {device} @ {width}x{height}\n"
                  f"{fps.mean():.2f} FPS end to end")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plot_path = os.path.join(output_dir, f"latency_breakdown_{device.type}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot : {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
