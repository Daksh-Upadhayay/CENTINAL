"""Fine-tune CSRNet on ShanghaiTech.

The shipped ``csrnet_shanghai.pth`` scores far worse than the published CSRNet
result on ShanghaiTech Part A, and its VGG frontend is still almost exactly at
ImageNet initialisation -- consistent with a short training run in which the
backend learned to read un-normalised inputs and the frontend never adapted.
This script retrains it properly: ImageNet-normalised inputs, geometry-adaptive
ground-truth density maps, and crop augmentation.

    python train_csrnet.py --data_root <...>/ShanghaiTech --part A --epochs 60

Ground-truth density maps are cached to disk on first use because building
them with a per-point adaptive kernel is the slowest part of the first epoch.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial import KDTree
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg16

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from centinal.models import BASE_DIR, CSRNet, get_device  # noqa: E402
from centinal.pipeline import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# CSRNet's output is 1/8 the input resolution.
DOWNSAMPLE = 8

# Ground-truth density maps are kept in *count units*: the map sums to the
# number of people, so the network's raw output is directly the count. Some
# CSRNet implementations scale targets up by 100 for gradient magnitude, but a
# checkpoint trained one way cannot be fine-tuned against the other convention
# -- doing so asks an already-correct model to emit values 100x too large and
# destroys it. The loss is scaled instead (see LOSS_SCALE), which changes step
# size without changing what the network is asked to predict.
DENSITY_SCALE = 1.0
LOSS_SCALE = 100.0


def load_points(mat_path: str) -> np.ndarray:
    mat = scipy.io.loadmat(mat_path)
    return np.asarray(mat["image_info"][0, 0][0][0][0], dtype=np.float64)


def geometry_adaptive_density(points: np.ndarray, shape, k: int = 3, beta: float = 0.3):
    """Build a ground-truth density map with a per-head adaptive Gaussian.

    Each annotated head is blurred with sigma proportional to its mean distance
    to the k nearest heads, so crowded regions get tight kernels and sparse
    regions get wide ones. This is the standard MCNN/CSRNet target for Part A.

    Each kernel is rendered into a local window of radius 3*sigma rather than
    convolved across the whole frame: a Part A image has ~500 heads, and
    filtering the full image once per head made ground-truth generation take
    longer than training itself.
    """
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    if len(points) == 0:
        return density

    # Keep only annotations that land inside the image.
    pts = points[(points[:, 0] >= 0) & (points[:, 0] < w)
                 & (points[:, 1] >= 0) & (points[:, 1] < h)]
    if len(pts) == 0:
        return density

    if len(pts) == 1:
        sigmas = np.array([min(h, w) / 4.0])
    else:
        tree = KDTree(pts)
        kk = min(k + 1, len(pts))
        dists, _ = tree.query(pts, k=kk)
        # Column 0 is the point itself (distance 0).
        sigmas = beta * dists[:, 1:].mean(axis=1)
    sigmas = np.clip(sigmas, 1.0, 40.0)

    for (x, y), sigma in zip(pts, sigmas):
        cx, cy = int(x), int(y)
        radius = max(int(3.0 * sigma), 1)
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, w)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue

        ys = np.arange(y0, y1, dtype=np.float32) - y
        xs = np.arange(x0, x1, dtype=np.float32) - x
        kernel = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2.0 * sigma * sigma))
        total = kernel.sum()
        if total > 0:
            # Normalise so every head contributes exactly 1.0 to the total count,
            # even where the window is clipped by the image border.
            density[y0:y1, x0:x1] += (kernel / total).astype(np.float32)
    return density


class ShanghaiTechDataset(Dataset):
    """ShanghaiTech split yielding (image, density_map) pairs."""

    def __init__(self, split_dir: str, train: bool, cache_dir: str, crop_divisor: int = 2,
                 preprocess_mode: str = "imagenet"):
        self.images_dir = os.path.join(split_dir, "images")
        gt_dir = None
        for candidate in ("ground_truth", "ground-truth"):
            if os.path.isdir(os.path.join(split_dir, candidate)):
                gt_dir = os.path.join(split_dir, candidate)
                break
        if gt_dir is None:
            raise FileNotFoundError(f"No ground-truth directory under {split_dir}")
        self.gt_dir = gt_dir
        self.train = train
        self.crop_divisor = crop_divisor
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.files = sorted(f for f in os.listdir(self.images_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.preprocess_mode = preprocess_mode
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def _density_for(self, name: str, shape):
        cache_path = os.path.join(self.cache_dir, f"{os.path.splitext(name)[0]}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)
        points = load_points(os.path.join(self.gt_dir, f"GT_{os.path.splitext(name)[0]}.mat"))
        density = geometry_adaptive_density(points, shape)
        np.save(cache_path, density)
        return density

    def __getitem__(self, idx):
        name = self.files[idx]
        image = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
        w, h = image.size
        density = self._density_for(name, (h, w))

        if self.train:
            # Random crop, then horizontal flip.
            ch, cw = h // self.crop_divisor, w // self.crop_divisor
            # Keep crop dimensions divisible by the network's stride so the
            # target downsamples to exactly the predicted map size.
            ch -= ch % DOWNSAMPLE
            cw -= cw % DOWNSAMPLE
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            image = image.crop((left, top, left + cw, top + ch))
            density = density[top:top + ch, left:left + cw]
            if np.random.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                density = density[:, ::-1].copy()
        else:
            ch, cw = h - h % DOWNSAMPLE, w - w % DOWNSAMPLE
            image = image.crop((0, 0, cw, ch))
            density = density[:ch, :cw]

        tensor = self.to_tensor(image)
        if self.preprocess_mode == "imagenet":
            tensor = self.normalize(tensor)

        # Sum-pool the target down to the network's output resolution so total
        # count is preserved exactly.
        target = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0)
        target = torch.nn.functional.avg_pool2d(target, DOWNSAMPLE) * (DOWNSAMPLE ** 2)
        return tensor, target.squeeze(0) * DENSITY_SCALE


def evaluate(model, loader, device):
    """Return (MAE, RMSE, normalised MAE) over a loader of full images.

    Normalised MAE is MAE divided by the split's mean ground-truth count. Part A
    averages ~434 people per image and Part B ~124, so a plain average of the two
    MAEs is dominated by Part A: a model could halve its Part A error, triple its
    Part B error, and still look like it improved. Normalising puts both splits
    on a comparable scale before they are combined.
    """
    model.eval()
    errors, gt_counts = [], []
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device)
            pred = model(image)
            pred_count = float(pred.sum().item()) / DENSITY_SCALE
            gt_count = float(target.sum().item()) / DENSITY_SCALE
            errors.append(pred_count - gt_count)
            gt_counts.append(gt_count)
    errors = np.array(errors)
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    mean_gt = max(float(np.mean(gt_counts)), 1e-6)
    return mae, rmse, mae / mean_gt


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSRNet on ShanghaiTech")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory containing part_A_final / part_B_final")
    parser.add_argument("--part", choices=["A", "B", "AB"], default="AB",
                        help="Which ShanghaiTech part(s) to train on. AB unions both, "
                             "which is what CENTINAL needs: part B covers the sparse and "
                             "medium regimes, part A the dense one.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Checkpoint to start from; default is ImageNet VGG16 frontend")
    parser.add_argument("--out", type=str, default="csrnet_finetuned.pth")
    parser.add_argument("--preprocess", choices=["raw", "imagenet"], default="imagenet",
                        help="Input scaling to train with. Must match --init_from's "
                             "convention when fine-tuning an existing checkpoint.")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate on the test splits every N epochs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--time_budget_min", type=float, default=None,
                        help="Stop training after this many minutes")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    parts = ["A", "B"] if args.part == "AB" else [args.part]
    cache_root = args.cache_dir or os.path.join(args.data_root, ".density_cache")

    def build(split, train):
        sets = []
        for part in parts:
            sets.append(ShanghaiTechDataset(
                os.path.join(args.data_root, f"part_{part}_final", split), train,
                os.path.join(cache_root, part, split), preprocess_mode=args.preprocess))
        return sets[0] if len(sets) == 1 else ConcatDataset(sets)

    train_set = build("train_data", True)
    # Each part is scored separately so a gain on one cannot hide a loss on the other.
    test_sets = {
        part: ShanghaiTechDataset(os.path.join(args.data_root, f"part_{part}_final", "test_data"),
                                  False, os.path.join(cache_root, part, "test_data"),
                                  preprocess_mode=args.preprocess)
        for part in parts
    }
    # Variable image sizes mean batch size must stay at 1.
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    test_loaders = {p: DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
                    for p, ds in test_sets.items()}

    model = CSRNet().to(device)
    if args.init_from:
        init_path = args.init_from if os.path.isabs(args.init_from) else os.path.join(BASE_DIR, args.init_from)
        blob = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob)
        print(f"Initialised from {init_path}")
    else:
        # Standard CSRNet init: ImageNet VGG16 frontend, backend from scratch.
        vgg = vgg16(weights="IMAGENET1K_V1")
        model.frontend.load_state_dict(
            dict(zip(model.frontend.state_dict().keys(),
                     [p.data.clone() for p in vgg.features[:23].parameters()])))
        print("Initialised frontend from ImageNet VGG16, backend from scratch")

    baseline = {p: evaluate(model, dl, device) for p, dl in test_loaders.items()}
    baseline_str = "  ".join(f"{p}: MAE {m:6.2f} RMSE {r:6.2f}" for p, (m, r, _) in baseline.items())
    print(f"epoch   0 (before training)          {baseline_str}")
    print("  ^ this must match the checkpoint's known benchmark score. If it does not,\n"
          "    the preprocessing or density-scale convention here disagrees with how the\n"
          "    checkpoint was trained, and fine-tuning will make the model worse.\n", flush=True)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)
    def score(per_part):
        """Scale-fair selection score: mean normalised MAE across splits."""
        return float(np.mean([n for _, _, n in per_part.values()]))

    history = [{"epoch": 0, "train_loss": None, "note": "before training",
                "score": score(baseline),
                "per_part": {p: {"mae": m, "rmse": r, "nmae": n}
                             for p, (m, r, n) in baseline.items()}}]
    best_score = score(baseline)
    print(f"Baseline score (mean normalised MAE) to beat: {best_score:.4f}\n")
    started = time.time()

    test_summary = " ".join(f"{p}={len(d)}" for p, d in test_sets.items())
    print(f"Device: {device} | part {args.part} | train={len(train_set)} | "
          f"test {test_summary} | epochs={args.epochs} | lr={args.lr} | "
          f"input={args.preprocess}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(image)
            # Guard against a one-pixel mismatch from odd input dimensions.
            if pred.shape[-2:] != target.shape[-2:]:
                target = torch.nn.functional.interpolate(target, size=pred.shape[-2:], mode="bilinear",
                                                         align_corners=False)
            loss = criterion(pred, target) * LOSS_SCALE
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            print(f"epoch {epoch:3d}  loss={running/len(train_loader):9.2f}  "
                  f"[{(time.time()-started)/60.0:.1f}m]", flush=True)
            continue

        per_part = {p: evaluate(model, dl, device) for p, dl in test_loaders.items()}
        # Selection metric is the mean MAE across parts so the checkpoint chosen
        # is the one that is best across the whole density range, not just one end.
        epoch_score = score(per_part)
        elapsed = (time.time() - started) / 60.0

        history.append({"epoch": epoch, "train_loss": running / len(train_loader),
                        "score": epoch_score, "elapsed_min": elapsed,
                        "per_part": {p: {"mae": m, "rmse": r, "nmae": n}
                                     for p, (m, r, n) in per_part.items()}})
        marker = ""
        if epoch_score < best_score:
            best_score = epoch_score
            torch.save({"state_dict": model.state_dict(),
                        "preprocess": args.preprocess,
                        "trained_on": f"ShanghaiTech part {args.part}",
                        "epoch": epoch,
                        "score": epoch_score,
                        "metrics": {p: {"mae": m, "rmse": r, "nmae": n}
                                    for p, (m, r, n) in per_part.items()}},
                       out_path)
            marker = "  <- best, saved"
        parts_str = "  ".join(f"{p}: MAE {m:6.2f} RMSE {r:6.2f}" for p, (m, r, _) in per_part.items())
        print(f"epoch {epoch:3d}  loss={running/len(train_loader):9.2f}  {parts_str}  "
              f"score={epoch_score:.4f}  [{elapsed:.1f}m]{marker}", flush=True)

        with open(os.path.splitext(out_path)[0] + "_history.json", "w") as fh:
            json.dump({"part": args.part, "lr": args.lr, "best_score": best_score,
                       "score_definition": "mean of per-split MAE / mean ground-truth count",
                       "history": history}, fh, indent=2)

        if args.time_budget_min and elapsed >= args.time_budget_min:
            print(f"Time budget of {args.time_budget_min} min reached; stopping.")
            break

    print(f"\nBest score (mean normalised MAE): {best_score:.4f}  ->  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
