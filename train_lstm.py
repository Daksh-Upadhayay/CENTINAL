"""Train the CENTINAL LSTM risk classifier.

Why this script exists
----------------------
The shipped ``risk_lstm.h5`` was trained with a 30-frame window, but there was
no training code in the repository, so nothing tied the weights to the feature
pipeline that produces them. When CSRNet preprocessing was corrected to use
ImageNet normalisation, the feature distribution shifted and the old scaler no
longer matched. This script regenerates the dataset and the scaler from the
current pipeline so the two can never drift apart silently again.

Labelling honesty
-----------------
There is no human-annotated stampede dataset here. Labels are derived from an
explicit, physically-motivated rule over the density signal (level, rate of
change, turbulence) -- see ``label_window``. The model therefore learns to
apply that rule with temporal context; it does not learn stampede risk from
observed stampedes. Metrics produced against these labels measure how well the
network reproduces the rule on held-out scenarios, and the README says so.

Crucially the rule reads only observable density features, never the model's
own output, so evaluation is not circular.

Calibration assumption
----------------------
CSRNet's count wobbles from frame to frame even when nothing in the scene
changes. "Turbulence" is therefore measured relative to that wobble, and the
wobble is measured on the calibration video. The calibration clip must show a
calm crowd: calibrating on footage of a genuine surge would teach the rule that
surging is normal.

Usage
-----
    python train_lstm.py --video videos/crowd_test.mp4
    python train_lstm.py --video videos/crowd_test.mp4 --epochs 80
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from centinal.models import BASE_DIR, load_csrnet  # noqa: E402
from centinal.pipeline import MEDIUM_THRESHOLD, SPARSE_THRESHOLD  # noqa: E402

SEQ_LEN = 30
N_FEATURES = 4
N_CLASSES = 3
RISK_NAMES = ("SAFE", "WARNING", "CRITICAL")

# A single camera view has a finite footprint, so crowd count saturates rather
# than growing without bound. Without this ceiling a sustained surge compounds
# geometrically into counts of several thousand -- values no real frame can
# produce, which would stretch the feature scaler's range and squeeze every
# realistic count into a narrow band of its output.
MAX_PLAUSIBLE_COUNT = 1200.0

# ---------------------------------------------------------------------------
# Labelling rule
# ---------------------------------------------------------------------------

# A crowd becomes dangerous through a combination of how packed it is, how fast
# it is packing further, and how erratically density is moving around the frame.
# These thresholds are expressed relative to the count scale so they stay
# meaningful across resolutions.
SURGE_RATE = 0.015        # fractional growth in count per frame

# A window is turbulent when its frame-to-frame variability is this many times
# what CSRNet's own counting noise produces on a static crowd. An absolute
# threshold does not work: the estimator's noise alone (about 4% of the count
# per frame on 1080p footage) sits above any small fixed cut-off, which made
# every dense crowd look turbulent and therefore CRITICAL.
TURBULENCE_RATIO = 1.6

# Fallback per-frame relative counting noise when calibration is skipped.
DEFAULT_MEASUREMENT_JITTER = 0.02


def window_dynamics(window: np.ndarray):
    """Summarise a feature window as (final_count, growth_rate, turbulence)."""
    counts = window[:, 0]
    deltas = window[:, 2]

    final_count = float(counts[-1])
    baseline = max(float(np.mean(counts)), 1.0)

    # Least-squares slope of count over the window, as a fraction of the mean.
    t = np.arange(len(counts), dtype=np.float64)
    slope = float(np.polyfit(t, counts, 1)[0])
    growth_rate = slope / baseline

    mean_density = max(float(np.mean(window[:, 1])), 1e-9)
    turbulence = float(np.std(deltas)) / mean_density

    return final_count, growth_rate, turbulence


def label_window(window: np.ndarray, measurement_jitter: float) -> int:
    """Assign a risk class to one feature window.

    ``measurement_jitter`` is CSRNet's per-frame relative counting noise. Pure
    noise of that size gives a frame-to-frame turbulence of jitter * sqrt(2),
    so that is the baseline turbulence is judged against.

    Returns 0 SAFE, 1 WARNING, 2 CRITICAL.
    """
    count, growth, turbulence = window_dynamics(window)
    noise_turbulence = measurement_jitter * np.sqrt(2.0)
    unstable = growth > SURGE_RATE or turbulence > TURBULENCE_RATIO * noise_turbulence

    if count < SPARSE_THRESHOLD:
        # Too few people present for a crush regardless of how they are moving.
        return 0
    if count < MEDIUM_THRESHOLD:
        return 1 if unstable else 0
    # Dense crowd: stable is already worth a warning, unstable is critical.
    return 2 if unstable else 1


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

# Crowd dynamics per scenario. Turbulence is the size of genuine random motion
# in the crowd, expressed as a multiple of the measured counting noise so that
# "calm" and "turbulent" stay distinguishable whatever the camera's noise level.
# Measurement noise itself is added separately to every scenario at the
# calibrated level, because it is a property of the estimator, not the crowd.
SCENARIOS = (
    # name,             base_count, trend_per_frame, turbulence (x noise)
    ("empty_plaza",           12.0,           0.000,   0.2),
    ("stable_sparse",         30.0,           0.000,   0.2),
    ("sparse_drift",          55.0,           0.002,   0.3),
    ("sparse_churn",          45.0,           0.000,   3.0),
    ("stable_medium",        160.0,           0.000,   0.2),
    ("stable_dense",         330.0,           0.000,   0.2),
    ("gradual_fill",          90.0,           0.008,   0.3),
    ("dispersal",            300.0,          -0.012,   0.4),
    ("medium_surge",         170.0,           0.022,   0.6),
    ("dense_surge",          290.0,           0.028,   0.6),
    ("turbulent_dense",      320.0,           0.002,   3.0),
    ("turbulent_medium",     150.0,           0.002,   3.0),
    ("sparse_influx",         40.0,           0.020,   0.5),
)


def synthesise_scenario(rng, base_count, trend, turbulence, jitter,
                        n_frames, density_pixels, variance_ratio):
    """Generate one physically consistent feature track.

    ``count`` and ``avg_density`` are not independent: CSRNet's count is the sum
    of the density map, so count == avg_density * number_of_density_pixels. The
    generator preserves that identity, otherwise the network would be trained on
    feature combinations the real pipeline can never produce.
    """
    counts = np.empty(n_frames, dtype=np.float64)
    level = base_count
    for i in range(n_frames):
        # Logistic damping: growth tapers as the view approaches capacity.
        headroom = max(1.0 - level / MAX_PLAUSIBLE_COUNT, 0.0)
        growth = trend * headroom + rng.normal(0.0, turbulence)
        level *= (1.0 + growth)
        level = float(np.clip(level, 1.0, MAX_PLAUSIBLE_COUNT))
        counts[i] = float(np.clip(level * (1.0 + rng.normal(0.0, jitter)),
                                  1.0, MAX_PLAUSIBLE_COUNT))

    avg_density = counts / density_pixels

    deltas = np.empty(n_frames, dtype=np.float64)
    deltas[0] = 0.0
    deltas[1:] = np.diff(avg_density)

    # Spatial variance scales with density and rises when the crowd is churning.
    churn = 1.0 + 6.0 * np.abs(deltas) / max(avg_density.mean(), 1e-9)
    spatial_var = variance_ratio * (avg_density ** 2) * churn
    spatial_var *= np.clip(1.0 + rng.normal(0.0, 0.10, n_frames), 0.3, None)

    return np.stack([counts, avg_density, deltas, spatial_var], axis=1)


def windows_from_track(track, seq_len, measurement_jitter, stride=1):
    """Slice a scenario track into overlapping windows plus their labels."""
    xs, ys = [], []
    for start in range(0, len(track) - seq_len + 1, stride):
        window = track[start:start + seq_len]
        xs.append(window)
        ys.append(label_window(window, measurement_jitter))
    return xs, ys


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def calibrate_from_video(video_path, max_frames, verbose=True):
    """Measure the real pipeline's feature scales so synthetic data matches it.

    Returns ``(density_pixels, variance_ratio, measurement_jitter, real_track)``.
    """
    from centinal.video import extract_video_features

    model, device, meta = load_csrnet()
    if verbose:
        print(f"Calibrating on {os.path.basename(video_path)} "
              f"(device: {device}, input: {meta['preprocess']})")
    track = extract_video_features(video_path, model, device, meta["preprocess"],
                                   max_frames=max_frames, verbose=verbose)

    counts, avg_density, _, spatial_var = (track[:, i] for i in range(4))
    density_pixels = float(np.median(counts / np.maximum(avg_density, 1e-12)))
    variance_ratio = float(np.median(spatial_var / np.maximum(avg_density ** 2, 1e-12)))

    # Per-frame counting noise: on a calm crowd, frame-to-frame changes are
    # dominated by estimator noise, and the std of differences of iid noise is
    # sqrt(2) times the noise itself.
    per_window = [np.std(np.diff(counts[i:i + SEQ_LEN])) / max(np.mean(counts[i:i + SEQ_LEN]), 1.0)
                  for i in range(0, max(len(counts) - SEQ_LEN + 1, 1))]
    measurement_jitter = float(np.median(per_window) / np.sqrt(2.0))

    if verbose:
        print(f"  frames={len(track)}  count range {counts.min():.0f}-{counts.max():.0f}")
        print(f"  density map pixels ~= {density_pixels:.0f}")
        print(f"  variance / density^2 ~= {variance_ratio:.3f}")
        print(f"  counting noise ~= {measurement_jitter * 100:.2f}% per frame")
    return density_pixels, variance_ratio, measurement_jitter, track


def build_dataset(rng, density_pixels, variance_ratio, measurement_jitter,
                  tracks_per_scenario, frames_per_track, seq_len):
    """Generate scenario tracks and split them so no track spans two splits.

    Overlapping windows from the same track are highly correlated. Splitting at
    the window level would leak nearly identical samples between train and test
    and inflate the reported scores, so the split is done per track.
    """
    per_split = {"train": [], "val": [], "test": []}

    for name, base, trend, turb_multiple in SCENARIOS:
        for i in range(tracks_per_scenario):
            track = synthesise_scenario(rng, base, trend, turb_multiple * measurement_jitter,
                                        measurement_jitter, frames_per_track,
                                        density_pixels, variance_ratio)
            # Deterministic 70/15/15 split over each scenario's tracks.
            frac = i / tracks_per_scenario
            split = "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")
            per_split[split].append(track)

    out = {}
    for split, tracks in per_split.items():
        xs, ys = [], []
        for track in tracks:
            wx, wy = windows_from_track(track, seq_len, measurement_jitter)
            xs.extend(wx)
            ys.extend(wy)
        out[split] = (np.stack(xs), np.array(ys, dtype=np.int64))
    return out


def main():
    parser = argparse.ArgumentParser(description="Train the CENTINAL LSTM risk classifier")
    parser.add_argument("--video", type=str, default="videos/crowd_test.mp4",
                        help="Video used to calibrate feature scales")
    parser.add_argument("--calib_frames", type=int, default=120,
                        help="Frames of video to run through CSRNet for calibration")
    parser.add_argument("--tracks", type=int, default=40,
                        help="Synthetic tracks generated per scenario")
    parser.add_argument("--track_frames", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_dir", type=str, default="lstm")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Use nominal 1080p feature scales instead of running CSRNet")
    args = parser.parse_args()

    import tensorflow as tf

    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    video_path = args.video if os.path.isabs(args.video) else os.path.join(BASE_DIR, args.video)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(BASE_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.skip_calibration or not os.path.exists(video_path):
        # 1920x1080 -> CSRNet density map is 1/8 scale -> 240 x 135 pixels.
        density_pixels, variance_ratio = 240.0 * 135.0, 0.35
        measurement_jitter = DEFAULT_MEASUREMENT_JITTER
        print(f"Skipping calibration; assuming {density_pixels:.0f} density pixels "
              f"and {measurement_jitter * 100:.1f}% counting noise.")
    else:
        density_pixels, variance_ratio, measurement_jitter, _ = calibrate_from_video(
            video_path, args.calib_frames)

    splits = build_dataset(rng, density_pixels, variance_ratio, measurement_jitter,
                           args.tracks, args.track_frames, SEQ_LEN)

    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]
    x_test, y_test = splits["test"]

    print("\nDataset (windows per split, class distribution):")
    for split, (x, y) in splits.items():
        dist = {RISK_NAMES[c]: int((y == c).sum()) for c in range(N_CLASSES)}
        print(f"  {split:<6} n={len(x):<6} {dist}")

    # The scaler is fitted on training windows only. Fitting it on the full
    # dataset would leak test-set range information into the inputs.
    scaler = MinMaxScaler()
    scaler.fit(x_train.reshape(-1, N_FEATURES))

    def scale(x):
        return scaler.transform(x.reshape(-1, N_FEATURES)).reshape(x.shape)

    xs_train, xs_val, xs_test = scale(x_train), scale(x_val), scale(x_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Classes are naturally imbalanced across the scenario mix; weight the loss
    # so CRITICAL is not traded away for overall accuracy.
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float64)
    class_weight = {i: float(len(y_train) / (N_CLASSES * c)) if c else 0.0
                    for i, c in enumerate(counts)}
    print(f"\nClass weights: { {RISK_NAMES[k]: round(v, 3) for k, v in class_weight.items()} }")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    model.fit(xs_train, y_train,
              validation_data=(xs_val, y_val),
              epochs=args.epochs, batch_size=args.batch_size,
              class_weight=class_weight, callbacks=callbacks, verbose=2)

    test_loss, test_acc = model.evaluate(xs_test, y_test, verbose=0)
    print(f"\nHeld-out test: loss={test_loss:.4f} accuracy={test_acc:.4f}")

    lstm_path = os.path.join(out_dir, "risk_lstm.h5")
    scaler_path = os.path.join(out_dir, "scaler.save")
    model.save(lstm_path)
    joblib.dump(scaler, scaler_path)

    # Persist the held-out split so eval_lstm.py scores the same data the model
    # never trained on, rather than regenerating a fresh sample.
    testset_path = os.path.join(out_dir, "risk_testset.npz")
    np.savez_compressed(testset_path, x=x_test, y=y_test)

    meta = {
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "classes": list(RISK_NAMES),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "density_pixels": density_pixels,
        "variance_ratio": variance_ratio,
        "max_plausible_count": MAX_PLAUSIBLE_COUNT,
        "measurement_jitter": measurement_jitter,
        "label_rule": {"surge_rate": SURGE_RATE, "turbulence_ratio": TURBULENCE_RATIO,
                       "sparse_threshold": SPARSE_THRESHOLD, "medium_threshold": MEDIUM_THRESHOLD},
        "labels": "rule-derived from density dynamics, not human annotation",
        "windows": {k: int(len(v[0])) for k, v in splits.items()},
        "test_accuracy": float(test_acc),
    }
    with open(os.path.join(out_dir, "training_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved model  : {lstm_path}")
    print(f"Saved scaler : {scaler_path}")
    print(f"Saved testset: {testset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
