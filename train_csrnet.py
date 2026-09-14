"""Fine-tune CSRNet on one or more crowd-counting datasets.

Supports ShanghaiTech A and B, UCF-QNRF and JHU-CROWD++ (see
``centinal/datasets.py`` for the layouts). The original checkpoint was weakest
on very large crowds, which ShanghaiTech barely covers: only 19 of its 498
test images hold more than 1,000 people. UCF-QNRF (up to 12,865 per image) and
JHU-CROWD++ (up to ~25,000, with weather and lighting variation) fill that gap.

    python train_csrnet.py --data_root /content/data --datasets sha,shb,qnrf,jhu \\
        --preprocess raw --init_from csrnet_centinal.pth

Checkpoint selection uses validation data only: ShanghaiTech and UCF-QNRF have
no official validation split, so a fixed fraction of each training set is held
out; JHU-CROWD++ ships one. Test sets are never consulted during training --
score them afterwards with eval/eval_csrnet.py.

Training runs on a GPU (see colab/train_csrnet_colab.ipynb).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial import KDTree
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from centinal.datasets import (  # noqa: E402
    DATASET_LABELS,
    DATASET_NAMES,
    MAX_SIDE,
    MIN_SIDE,
    dataset_splits,
    holdout,
    load_image_and_points,
)
from centinal.models import BASE_DIR, CSRNet, get_device  # noqa: E402
from centinal.pipeline import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# CSRNet's output is 1/8 the input resolution.
DOWNSAMPLE = 8

# Ground-truth density maps are kept in *count units*: the map sums to the
# number of people, so the network's raw output is directly the count. Some
# CSRNet implementations scale targets up by 100 for gradient magnitude, but a
# checkpoint trained one way cannot be fine-tuned against the other convention
# -- doing so asks an already-correct model to emit values 100x too large and
# destroys it. The loss is scaled instead, which changes step size without
# changing what the network is asked to predict.
LOSS_SCALE = 100.0

# Epoch-0 check: if the starting checkpoint scores more than this far from the
# result recorded when it was trained, the setup here disagrees with how it was
# trained and fine-tuning would degrade it.
BASELINE_TOLERANCE = 0.10

# Recorded ShanghaiTech scores refer to the complete official test sets.
OFFICIAL_TEST_SIZES = {"sha": 182, "shb": 316}


def geometry_adaptive_density(points: np.ndarray, shape, k: int = 3, beta: float = 0.3):
    """Build a ground-truth density map with a per-head adaptive Gaussian.

    Each annotated head is blurred with sigma proportional to its mean distance
    to the k nearest heads, so crowded regions get tight kernels and sparse
    regions get wide ones. This is the standard MCNN/CSRNet target.

    Heads annotated fractionally outside the frame -- common at image borders in
    UCF-QNRF -- are clamped onto the edge rather than dropped, so the map always
    sums to the annotated count. Each kernel is rendered into a local window of
    radius 3*sigma: filtering the whole frame once per head made ground-truth
    generation slower than training itself.
    """
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density

    pts = np.column_stack([np.clip(points[:, 0], 0, w - 1), np.clip(points[:, 1], 0, h - 1)])

    if len(pts) == 1:
        sigmas = np.array([min(h, w) / 4.0])
    else:
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=min(k + 1, len(pts)))
        # Column 0 is the point itself (distance 0).
        sigmas = beta * dists[:, 1:].mean(axis=1)
    sigmas = np.clip(sigmas, 1.0, 40.0)

    for (x, y), sigma in zip(pts, sigmas):
        cx, cy = int(x), int(y)
        radius = max(int(3.0 * sigma), 1)
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
        ys = np.arange(y0, y1, dtype=np.float32) - y
        xs = np.arange(x0, x1, dtype=np.float32) - x
        kernel = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * sigma * sigma))
        total = kernel.sum()
        if total > 0:
            # Every head contributes exactly 1.0, even where the window is clipped.
            density[y0:y1, x0:x1] += (kernel / total).astype(np.float32)
    return density


class CrowdDataset(Dataset):
    """Yields (image tensor, target density at 1/8 resolution) pairs.

    Targets are cached already sum-pooled to the network's output resolution.
    Crops are aligned to the 8-pixel grid so a crop of the image corresponds
    exactly to a crop of the pooled target. Caching full-resolution maps
    instead would need tens of gigabytes for UCF-QNRF and JHU-CROWD++.
    """

    def __init__(self, samples, train: bool, cache_root: str,
                 preprocess_mode: str = "imagenet", crop_divisor: int = 2):
        self.samples = list(samples)
        self.train = train
        self.cache_root = os.path.join(cache_root, f"s{MAX_SIDE}-{MIN_SIDE}")
        self.crop_divisor = crop_divisor
        self.preprocess_mode = preprocess_mode
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def _target(self, sample, points, shape):
        path = os.path.join(self.cache_root, sample.key + ".npy")
        if os.path.exists(path):
            cached = np.load(path)
            if cached.shape == (shape[0] // DOWNSAMPLE, shape[1] // DOWNSAMPLE):
                return cached
        h, w = shape
        density = geometry_adaptive_density(points, shape)
        pooled = density.reshape(h // DOWNSAMPLE, DOWNSAMPLE, w // DOWNSAMPLE, DOWNSAMPLE).sum(axis=(1, 3))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, pooled.astype(np.float32))
        return pooled.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image, points = load_image_and_points(sample)
        w, h = image.size
        hc, wc = h - h % DOWNSAMPLE, w - w % DOWNSAMPLE
        image = image.crop((0, 0, wc, hc))
        target = self._target(sample, points, (hc, wc))

        if self.train:
            ch = max(DOWNSAMPLE, (hc // self.crop_divisor) // DOWNSAMPLE * DOWNSAMPLE)
            cw = max(DOWNSAMPLE, (wc // self.crop_divisor) // DOWNSAMPLE * DOWNSAMPLE)
            top = DOWNSAMPLE * np.random.randint(0, (hc - ch) // DOWNSAMPLE + 1)
            left = DOWNSAMPLE * np.random.randint(0, (wc - cw) // DOWNSAMPLE + 1)
            image = image.crop((left, top, left + cw, top + ch))
            target = target[top // DOWNSAMPLE:(top + ch) // DOWNSAMPLE,
                            left // DOWNSAMPLE:(left + cw) // DOWNSAMPLE]
            if np.random.rand() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target = target[:, ::-1]

        tensor = self.to_tensor(image)
        if self.preprocess_mode == "imagenet":
            tensor = self.normalize(tensor)
        return tensor, torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0)


def evaluate(model, loader, device):
    """Return (MAE, RMSE, normalised MAE) over a loader of full images.

    Normalised MAE is MAE divided by the split's mean ground-truth count.
    Datasets differ in typical crowd size by an order of magnitude, so a plain
    average of their MAEs is dominated by the densest one: a model could halve
    its error on UCF-QNRF, triple it on ShanghaiTech B, and still look better.
    """
    model.eval()
    errors, gt_counts = [], []
    with torch.no_grad():
        for image, target in loader:
            pred = model(image.to(device))
            pred_count = float(pred.sum().item())
            gt_count = float(target.sum().item())
            errors.append(pred_count - gt_count)
            gt_counts.append(gt_count)
    errors = np.array(errors)
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    # Floor of one person: a split made only of empty images would otherwise divide by zero.
    return mae, rmse, mae / max(float(np.mean(gt_counts)), 1.0)


def seed_worker(_):
    # DataLoader workers inherit identical NumPy RNG state; without this every
    # worker would draw the same random crops.
    np.random.seed(torch.initial_seed() % 2 ** 32)


def build_splits(args):
    """Return ({name: train samples}, {name: val samples}, {name: test samples})."""
    train, val, test = {}, {}, {}
    for name in args.datasets:
        splits = dataset_splits(args.data_root, name)
        if "val" in splits:
            train[name], val[name] = splits["train"], splits["val"]
        else:
            # Note: a checkpoint previously fine-tuned on the full training set has
            # already seen these held-out images, so its starting validation score
            # on them is optimistic. Datasets new to the model are unaffected.
            train[name], val[name] = holdout(splits["train"], args.val_fraction, args.seed)
        test[name] = splits["test"]
        if args.max_val_per_dataset and len(val[name]) > args.max_val_per_dataset:
            order = np.random.default_rng(args.seed).permutation(len(val[name]))[:args.max_val_per_dataset]
            val[name] = [val[name][i] for i in sorted(order)]
        if args.max_train_per_dataset:
            train[name] = train[name][:args.max_train_per_dataset]
    return train, val, test


def check_baseline(meta, model, device, args, test, val_scores, make_loader):
    """Stop early if the starting checkpoint does not reproduce its recorded score."""
    comparisons = []
    if "metrics" in meta and set(meta["metrics"]) <= {"A", "B"}:
        # Checkpoints from the ShanghaiTech-only trainer recorded test-set scores.
        for part, name in (("A", "sha"), ("B", "shb")):
            if part in meta["metrics"] and name in test:
                if len(test[name]) != OFFICIAL_TEST_SIZES[name]:
                    print(f"  ({DATASET_LABELS[name]} test set is incomplete; skipping its check)")
                    continue
                mae, _, _ = evaluate(model, make_loader(test[name]), device)
                comparisons.append((f"{DATASET_LABELS[name]} test", meta["metrics"][part]["mae"], mae))
    elif "val_metrics" in meta and meta.get("split_config") == split_config(args):
        for name, recorded in meta["val_metrics"].items():
            if name in val_scores:
                comparisons.append((f"{DATASET_LABELS[name]} val", recorded["mae"], val_scores[name][0]))

    if not comparisons:
        print("  (starting checkpoint has no comparable recorded score; skipping the check)")
        return
    for label, recorded, measured in comparisons:
        drift = abs(measured - recorded) / max(recorded, 1e-6)
        status = "ok" if drift <= BASELINE_TOLERANCE else "MISMATCH"
        print(f"  {label:<22} recorded MAE {recorded:8.2f}  measured {measured:8.2f}  [{status}]")
        if drift > BASELINE_TOLERANCE:
            raise SystemExit(
                f"\nThe starting checkpoint scores {drift:.0%} away from its recorded result on "
                f"{label}. The preprocessing or density conventions here disagree with how it "
                "was trained, and fine-tuning would make it worse. Check --preprocess.")


def split_config(args):
    return {"datasets": list(args.datasets), "val_fraction": args.val_fraction,
            "max_val_per_dataset": args.max_val_per_dataset, "seed": args.seed,
            "max_side": MAX_SIDE, "min_side": MIN_SIDE}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet on crowd-counting datasets")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory the datasets were extracted under (searched a few levels deep)")
    parser.add_argument("--datasets", type=str, default="sha,shb",
                        help=f"Comma-separated subset of {','.join(DATASET_NAMES)}")
    parser.add_argument("--part", choices=["A", "B", "AB"], default=None,
                        help="Deprecated: ShanghaiTech part(s). Use --datasets instead.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Checkpoint to start from; default is ImageNet VGG16 frontend")
    parser.add_argument("--out", type=str, default="csrnet_finetuned.pth")
    parser.add_argument("--preprocess", choices=["raw", "imagenet"], default="imagenet",
                        help="Input scaling to train with. Must match --init_from's convention.")
    parser.add_argument("--eval_every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Share of the training set held out for validation where a dataset "
                             "has no official validation split")
    parser.add_argument("--max_val_per_dataset", type=int, default=150,
                        help="Cap on validation images per dataset, to bound validation time")
    parser.add_argument("--max_train_per_dataset", type=int, default=None,
                        help="Only for quick pipeline checks")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--time_budget_min", type=float, default=None,
                        help="Stop training after this many minutes")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.part:
        args.datasets = {"A": "sha", "B": "shb", "AB": "sha,shb"}[args.part]
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = set(args.datasets) - set(DATASET_NAMES)
    if unknown:
        parser.error(f"unknown dataset(s) {sorted(unknown)}; choose from {DATASET_NAMES}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    cache_root = args.cache_dir or os.path.join(args.data_root, ".density_cache")

    train, val, test = build_splits(args)
    train_set = ConcatDataset([CrowdDataset(train[n], True, cache_root, args.preprocess)
                               for n in args.datasets])

    def make_loader(samples):
        return DataLoader(CrowdDataset(samples, False, cache_root, args.preprocess),
                          batch_size=1, shuffle=False, num_workers=args.workers,
                          worker_init_fn=seed_worker)

    # Variable image sizes mean batch size must stay at 1.
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.workers,
                              worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
    val_loaders = {n: make_loader(val[n]) for n in args.datasets}

    print(f"Device: {device} | lr={args.lr} | input={args.preprocess} | "
          f"images resized into [{MIN_SIDE}, {MAX_SIDE}] px")
    for n in args.datasets:
        print(f"  {DATASET_LABELS[n]:<15} train {len(train[n]):5d}  val {len(val[n]):4d}  "
              f"test {len(test[n]):5d} (test not used during training)")

    model = CSRNet().to(device)
    meta = {}
    if args.init_from:
        init_path = args.init_from if os.path.isabs(args.init_from) else os.path.join(BASE_DIR, args.init_from)
        blob = torch.load(init_path, map_location=device, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob:
            model.load_state_dict(blob["state_dict"])
            meta = {k: v for k, v in blob.items() if k != "state_dict"}
            if meta.get("preprocess") and meta["preprocess"] != args.preprocess:
                parser.error(f"{init_path} was trained with --preprocess {meta['preprocess']}, "
                             f"not {args.preprocess}")
        else:
            model.load_state_dict(blob)
        print(f"Initialised from {init_path}")
    else:
        # Standard CSRNet init: ImageNet VGG16 frontend, backend from scratch.
        vgg = vgg16(weights="IMAGENET1K_V1")
        model.frontend.load_state_dict(
            dict(zip(model.frontend.state_dict().keys(),
                     [p.data.clone() for p in vgg.features[:23].parameters()])))
        print("Initialised frontend from ImageNet VGG16, backend from scratch")

    def score(per_set):
        """Scale-fair selection score: mean normalised MAE across datasets."""
        return float(np.mean([nmae for _, _, nmae in per_set.values()]))

    def fmt(per_set):
        return "  ".join(f"{n}: MAE {m:7.2f}" for n, (m, _, _) in per_set.items())

    print("\nepoch 0 -- starting checkpoint, before any training:")
    baseline = {n: evaluate(model, dl, device) for n, dl in val_loaders.items()}
    print(f"  validation  {fmt(baseline)}  score={score(baseline):.4f}")
    check_baseline(meta, model, device, args, test, baseline, make_loader)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
    history_path = os.path.splitext(out_path)[0] + "_history.json"
    history = [{"epoch": 0, "note": "starting checkpoint", "score": score(baseline),
                "val": {n: {"mae": m, "rmse": r, "nmae": x} for n, (m, r, x) in baseline.items()}}]
    best_score = score(baseline)
    print(f"Score to beat (mean normalised validation MAE): {best_score:.4f}\n", flush=True)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(image)
            loss = criterion(pred, target) * LOSS_SCALE
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        elapsed = (time.time() - started) / 60.0
        mean_loss = running / len(train_loader)

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            print(f"epoch {epoch:3d}  loss={mean_loss:10.2f}  [{elapsed:.1f}m]", flush=True)
        else:
            per_set = {n: evaluate(model, dl, device) for n, dl in val_loaders.items()}
            epoch_score = score(per_set)
            elapsed = (time.time() - started) / 60.0
            history.append({"epoch": epoch, "train_loss": mean_loss, "score": epoch_score,
                            "elapsed_min": elapsed,
                            "val": {n: {"mae": m, "rmse": r, "nmae": x}
                                    for n, (m, r, x) in per_set.items()}})
            marker = ""
            if epoch_score < best_score:
                best_score = epoch_score
                torch.save({"state_dict": model.state_dict(),
                            "preprocess": args.preprocess,
                            "trained_on": [DATASET_LABELS[n] for n in args.datasets],
                            "epoch": epoch,
                            "score": epoch_score,
                            "split_config": split_config(args),
                            "val_metrics": {n: {"mae": m, "rmse": r, "nmae": x}
                                            for n, (m, r, x) in per_set.items()}},
                           out_path)
                marker = "  <- best, saved"
            print(f"epoch {epoch:3d}  loss={mean_loss:10.2f}  {fmt(per_set)}  "
                  f"score={epoch_score:.4f}  [{elapsed:.1f}m]{marker}", flush=True)
            with open(history_path, "w") as fh:
                json.dump({"datasets": args.datasets, "lr": args.lr, "best_score": best_score,
                           "score_definition": "mean over datasets of validation MAE / mean count",
                           "split_config": split_config(args), "history": history}, fh, indent=2)

        if args.time_budget_min and elapsed >= args.time_budget_min:
            print(f"Time budget of {args.time_budget_min} min reached; stopping.")
            break

    if best_score < history[0]["score"]:
        print(f"\nBest validation score {best_score:.4f} (started at {history[0]['score']:.4f}) -> {out_path}")
    else:
        print(f"\nNo epoch beat the starting checkpoint ({history[0]['score']:.4f}); nothing was saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
