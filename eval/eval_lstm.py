"""Evaluate the LSTM risk classifier.

Modes
-----
holdout  (default) Score the model on the held-out scenario split written by
         train_lstm.py. These are windows from scenario tracks the network
         never trained on, so the numbers are a genuine generalisation
         estimate -- against rule-derived labels, not human annotation.

manual   Score against your own frame-level labels for a real video.

video    No labels: report the risk decisions the pipeline produces over a
         video, plus confidence statistics. Diagnostic only, no accuracy.

The previous "synthetic" mode has been removed. It compared the LSTM against
labels that were themselves computed from the LSTM's own prediction, so it
could only ever report near-perfect agreement and measured nothing.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centinal.models import BASE_DIR, load_csrnet, load_risk_head  # noqa: E402
from centinal.pipeline import (  # noqa: E402
    RISK_LABELS,
    FeatureExtractor,
    RiskEngine,
    classify_scene,
    compute_risk,
    preprocess,
)

TARGET_NAMES = list(RISK_LABELS)


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
           title=title, ylabel="Ground truth", xlabel="Predicted")
    threshold = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def report(y_true, y_pred, confidences, title, output_dir, tag):
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    labels = [0, 1, 2]
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    # Classes are imbalanced, so macro F1 matters more than raw accuracy here.
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES,
                                labels=labels, zero_division=0, digits=3))
    print(f"  Accuracy            : {accuracy:.4f}")
    print(f"  Macro F1            : {macro_f1:.4f}")
    if len(confidences):
        print(f"  Mean confidence     : {np.mean(confidences):.4f}")
        correct = np.array(y_true) == np.array(y_pred)
        if correct.any():
            print(f"  Mean conf (correct) : {np.mean(np.asarray(confidences)[correct]):.4f}")
        if (~correct).any():
            print(f"  Mean conf (wrong)   : {np.mean(np.asarray(confidences)[~correct]):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"\n  Confusion matrix (rows = truth):\n{cm}")
    print("=" * 62)

    plot_path = os.path.join(output_dir, f"confusion_matrix_{tag}.png")
    plot_confusion(cm, f"{title}\nAccuracy {accuracy:.1%} | Macro F1 {macro_f1:.3f}", plot_path)
    print(f"  Confusion matrix plot: {plot_path}")

    metrics_path = os.path.join(output_dir, f"lstm_metrics_{tag}.json")
    with open(metrics_path, "w") as fh:
        json.dump({
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "per_class": classification_report(y_true, y_pred, target_names=TARGET_NAMES,
                                               labels=labels, zero_division=0,
                                               output_dict=True),
            "confusion_matrix": cm.tolist(),
            "n_samples": int(len(y_true)),
        }, fh, indent=2)
    print(f"  Metrics JSON         : {metrics_path}")


def run_holdout(args, lstm, scaler, seq_len, output_dir):
    testset_path = os.path.join(BASE_DIR, "lstm", "risk_testset.npz")
    if not os.path.exists(testset_path):
        print(f"Error: {testset_path} not found. Run train_lstm.py first.")
        return 1

    data = np.load(testset_path)
    x, y_true = data["x"], data["y"]
    if x.shape[1] != seq_len:
        print(f"Error: held-out windows are {x.shape[1]} frames but the model expects {seq_len}.")
        return 1

    scaled = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    probs = lstm.predict(scaled, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    report(y_true, y_pred, confidences,
           "LSTM RISK CLASSIFIER - HELD-OUT SCENARIOS", output_dir, "holdout")

    pd.DataFrame({
        "Ground_Truth": y_true,
        "Prediction": y_pred,
        "Confidence": confidences,
        "Final_Count": x[:, -1, 0],
    }).to_csv(os.path.join(output_dir, "lstm_results_holdout.csv"), index=False)
    return 0


def run_video(args, lstm, scaler, seq_len, output_dir):
    """Run the full pipeline over a video, optionally scoring against manual labels."""
    import cv2
    import torch

    video_path = args.video if os.path.isabs(args.video) else os.path.join(BASE_DIR, args.video)
    csrnet, device, meta = load_csrnet()
    mode = meta["preprocess"]

    manual = None
    if args.mode == "manual":
        labels_path = args.labels if os.path.isabs(args.labels) else os.path.join(BASE_DIR, args.labels)
        if not os.path.exists(labels_path):
            print(f"Error: labels file not found: {labels_path}")
            return 1
        manual = pd.read_csv(labels_path).set_index("frame_idx")["label"].to_dict()
        print(f"Loaded {len(manual)} manual labels.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return 1

    print(f"Device: {device} | input: {mode} | LSTM window: {seq_len}")
    extractor = FeatureExtractor()
    engine = RiskEngine(lstm, scaler, seq_len)

    rows, y_true, y_pred, confidences = [], [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.frames is not None and frame_idx >= args.frames):
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            density = csrnet(preprocess(rgb, device, mode)).squeeze().cpu().numpy()

        feats = extractor.extract(density)
        scene = classify_scene(feats.count)
        engine.push(feats)
        pred_idx, confidence = engine.predict()
        risk_idx = compute_risk(scene, pred_idx, confidence, engine.ready)

        if engine.ready:
            rows.append({"Frame_Idx": frame_idx, "Crowd_Count": round(feats.count, 1),
                         "Scene_Type": scene, "LSTM_Pred": pred_idx,
                         "LSTM_Confidence": round(confidence, 4),
                         "Pipeline_Risk": RISK_LABELS[risk_idx]})
            if manual is not None and frame_idx in manual:
                y_true.append(int(manual[frame_idx]))
                y_pred.append(pred_idx)
                confidences.append(confidence)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx} frames", flush=True)
    cap.release()

    if not rows:
        print(f"No windows produced; the video is shorter than the {seq_len}-frame LSTM window.")
        return 1

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"lstm_results_{args.mode}.csv")
    df.to_csv(csv_path, index=False)

    if manual is not None:
        if not y_true:
            print("None of the manual labels matched a frame with a full LSTM window.")
            return 1
        report(y_true, y_pred, confidences,
               "LSTM RISK CLASSIFIER - MANUAL LABELS", output_dir, "manual")
    else:
        print("\n" + "=" * 62)
        print("  PIPELINE BEHAVIOUR (no labels - diagnostic only)")
        print("=" * 62)
        print(f"  Frames with a full window : {len(df)}")
        print(f"  Crowd count               : {df.Crowd_Count.min():.0f} - {df.Crowd_Count.max():.0f} "
              f"(mean {df.Crowd_Count.mean():.0f})")
        print(f"  Scene distribution        : {df.Scene_Type.value_counts().to_dict()}")
        print(f"  LSTM class distribution   : "
              f"{ {RISK_LABELS[k]: int((df.LSTM_Pred == k).sum()) for k in range(3)} }")
        print(f"  Pipeline risk distribution: {df.Pipeline_Risk.value_counts().to_dict()}")
        print(f"  Confidence                : mean {df.LSTM_Confidence.mean():.3f}, "
              f"min {df.LSTM_Confidence.min():.3f}, max {df.LSTM_Confidence.max():.3f}")
        print("=" * 62)
    print(f"  Frame log: {csv_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate the LSTM risk classifier")
    parser.add_argument("--mode", choices=["holdout", "manual", "video"], default="holdout")
    parser.add_argument("--video", type=str, default="videos/crowd_test.mp4")
    parser.add_argument("--labels", type=str, default=None,
                        help="CSV with frame_idx,label columns (manual mode)")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="eval/results")
    args = parser.parse_args()

    if args.mode == "manual" and not args.labels:
        parser.error("--labels is required in manual mode")

    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    lstm, scaler, seq_len = load_risk_head()
    print(f"LSTM window: {seq_len} frames (read from the trained model)")

    if args.mode == "holdout":
        return run_holdout(args, lstm, scaler, seq_len, output_dir)
    return run_video(args, lstm, scaler, seq_len, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
