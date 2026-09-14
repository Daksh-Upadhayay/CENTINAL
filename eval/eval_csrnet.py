"""Evaluate CSRNet crowd-counting accuracy on a benchmark test split.

Reports the standard counting metrics (MAE, RMSE) alongside MAPE and GAME,
a localisation-aware metric that catches a model getting the right total for
the wrong reasons. Point it at a test folder from ShanghaiTech, UCF-QNRF or
JHU-CROWD++; the layout is detected automatically and images are resized with
the same policy used in training (see centinal/datasets.py).

    python eval/eval_csrnet.py --dataset_path <...>/part_A_final/test_data
    python eval/eval_csrnet.py --dataset_path <...>/UCF-QNRF_ECCV18/Test
    python eval/eval_csrnet.py --dataset_path <...>/jhu_crowd_v2.0/test

Pass --preprocess raw to reproduce the un-normalised preprocessing the
pipeline used before ImageNet normalisation was applied.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centinal.datasets import load_image_and_points, samples_for_split_dir  # noqa: E402
from centinal.models import BASE_DIR, load_csrnet  # noqa: E402
from centinal.pipeline import preprocess  # noqa: E402


def game(pred_density: np.ndarray, gt_points: np.ndarray, image_shape, level: int) -> float:
    """Grid Average Mean Error at the given level.

    Splits the frame into a 4^level grid and sums the absolute count error per
    cell, so predictions that place people in the wrong region are penalised
    even when the global total happens to match. GAME(0) is equivalent to
    absolute count error.

    ``gt_points`` are in image coordinates while ``pred_density`` is the
    downscaled CSRNet output, so each is binned against its own dimensions.
    """
    side = 2 ** level
    dh, dw = pred_density.shape
    img_h, img_w = image_shape
    ys, xs = gt_points[:, 1], gt_points[:, 0]

    total = 0.0
    for i in range(side):
        for j in range(side):
            r0, r1 = int(dh * i / side), int(dh * (i + 1) / side)
            c0, c1 = int(dw * j / side), int(dw * (j + 1) / side)
            pred_cell = pred_density[r0:r1, c0:c1].sum()

            gt_cell = np.count_nonzero(
                (ys >= img_h * i / side) & (ys < img_h * (i + 1) / side)
                & (xs >= img_w * j / side) & (xs < img_w * (j + 1) / side)
            )
            total += abs(pred_cell - gt_cell)
    return float(total)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CSRNet on ShanghaiTech")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to a ShanghaiTech split, e.g. part_A_final/test_data")
    parser.add_argument("--model_path", type=str, default="csrnet_centinal.pth")
    parser.add_argument("--output_dir", type=str, default="eval/results")
    parser.add_argument("--preprocess", choices=["imagenet", "raw"], default=None,
                        help="Override the input scaling. Defaults to whatever the checkpoint declares.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for output filenames, e.g. partA")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    args = parser.parse_args()

    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(BASE_DIR, args.model_path)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        kind, samples = samples_for_split_dir(args.dataset_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print(f"No annotated images found in {args.dataset_path}")
        return 1

    model, device, meta = load_csrnet(model_path)
    mode = args.preprocess or meta["preprocess"]
    print(f"Device: {device} | preprocessing: {mode} | layout: {kind} | images: {len(samples)}")

    rows = []
    inference_ms = []
    for idx, sample in enumerate(samples):
        pil_image, points = load_image_and_points(sample)
        gt_count = len(points)
        image = np.array(pil_image)
        tensor = preprocess(image, device, mode)

        start = time.perf_counter()
        with torch.no_grad():
            density_map = model(tensor)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms.append((time.perf_counter() - start) * 1000)

        density = density_map.squeeze().cpu().numpy()
        pred_count = float(density.sum())
        image_shape = image.shape[:2]

        rows.append({
            "Image": os.path.basename(sample.image_path),
            "Ground_Truth": gt_count,
            "Prediction": pred_count,
            "Absolute_Error": abs(pred_count - gt_count),
            "Squared_Error": (pred_count - gt_count) ** 2,
            # Undefined for images with nobody in them (JHU-CROWD++ has a few).
            "Percentage_Error": abs(pred_count - gt_count) / gt_count * 100 if gt_count else np.nan,
            "GAME_1": game(density, points, image_shape, 1),
            "GAME_2": game(density, points, image_shape, 2),
        })

        if (idx + 1) % 40 == 0 or (idx + 1) == len(samples):
            print(f"  [{idx + 1}/{len(samples)}]", flush=True)

    if not rows:
        print("No images evaluated.")
        return 1

    df = pd.DataFrame(rows)
    mae = df["Absolute_Error"].mean()
    rmse = float(np.sqrt(df["Squared_Error"].mean()))
    mape = df["Percentage_Error"].mean()

    tag = args.tag or os.path.basename(os.path.normpath(args.dataset_path))
    suffix = f"{tag}_{mode}"
    csv_path = os.path.join(output_dir, f"csrnet_results_{suffix}.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 56)
    print(f"  CSRNET ACCURACY - {tag} ({mode} preprocessing)")
    print("=" * 56)
    print(f"  Images evaluated : {len(df)}")
    print(f"  MAE              : {mae:.2f}")
    print(f"  RMSE             : {rmse:.2f}")
    print(f"  MAPE             : {mape:.1f}%")
    print(f"  GAME(1)          : {df['GAME_1'].mean():.2f}")
    print(f"  GAME(2)          : {df['GAME_2'].mean():.2f}")
    print(f"  Inference        : {np.mean(inference_ms):.0f} ms/image ({1000/np.mean(inference_ms):.2f} img/s)")
    print(f"  Saved            : {csv_path}")
    print("=" * 56)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 7))
        plt.scatter(df["Ground_Truth"], df["Prediction"], alpha=0.6, edgecolors="k", linewidths=0.4)
        hi = max(df["Ground_Truth"].max(), df["Prediction"].max())
        plt.plot([0, hi], [0, hi], "r--", label="Perfect (y = x)")
        plt.xlabel("Ground truth count")
        plt.ylabel("Predicted count")
        plt.title(f"CSRNet - {tag} ({mode})\nMAE {mae:.2f} | RMSE {rmse:.2f}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plot_path = os.path.join(output_dir, f"csrnet_scatter_{suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Scatter plot     : {plot_path}")
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
