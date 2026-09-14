"""Crowd-counting dataset discovery, annotation loading and image sizing.

Training and evaluation both read datasets through this module so that a
given image is always resized and annotated the same way, whichever script
touches it. Three layouts are supported:

  ShanghaiTech   part_{A,B}_final/{train,test}_data/{images,ground_truth}/
                 IMG_1.jpg + GT_IMG_1.mat   (annotation folder may be ground-truth)
  UCF-QNRF       UCF-QNRF_ECCV18/{Train,Test}/img_0001.jpg + img_0001_ann.mat
  JHU-CROWD++    jhu_crowd_v2.0/{train,val,test}/{images,gt}/0001.jpg + 0001.txt
                 (each txt row: x y w h occlusion blur; empty when nobody is present)
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from PIL import Image

# Image size policy, following the preprocessing the JHU-CROWD++ authors ship
# with their dataset: images larger than MAX_SIDE are shrunk and images smaller
# than MIN_SIDE are enlarged. UCF-QNRF averages around 2000x3000 pixels, which
# neither fits a free-tier GPU nor resembles CENTINAL's 1080p cameras.
# ShanghaiTech images fall inside the bounds and are left untouched, so its
# benchmark numbers stay comparable with earlier results.
MAX_SIDE = 2048
MIN_SIDE = 512

DATASET_NAMES = ("sha", "shb", "qnrf", "jhu")
DATASET_LABELS = {
    "sha": "ShanghaiTech A",
    "shb": "ShanghaiTech B",
    "qnrf": "UCF-QNRF",
    "jhu": "JHU-CROWD++",
}


@dataclass(frozen=True)
class Sample:
    """One annotated image. ``key`` is unique across datasets and splits."""

    image_path: str
    annotation_path: str
    kind: str  # "shanghaitech" | "qnrf" | "jhu"
    key: str


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def _points_from_mat(path: str, preferred_key: Optional[str]) -> np.ndarray:
    mat = scipy.io.loadmat(path)
    if preferred_key and preferred_key in mat:
        return np.asarray(mat[preferred_key], dtype=np.float64).reshape(-1, 2)
    # Fall back to the first Nx2 numeric array in the file.
    for key, value in mat.items():
        if not key.startswith("__") and isinstance(value, np.ndarray) \
                and value.ndim == 2 and value.shape[1] == 2 and value.dtype.kind in "fiu":
            return value.astype(np.float64)
    raise ValueError(f"No Nx2 point array found in {path}")


def load_points(sample: Sample) -> np.ndarray:
    """Return an (N, 2) array of head positions as (x, y) in original pixels."""
    if sample.kind == "shanghaitech":
        mat = scipy.io.loadmat(sample.annotation_path)
        return np.asarray(mat["image_info"][0, 0][0][0][0], dtype=np.float64).reshape(-1, 2)
    if sample.kind == "qnrf":
        return _points_from_mat(sample.annotation_path, "annPoints")
    if sample.kind == "jhu":
        if os.path.getsize(sample.annotation_path) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        rows = np.loadtxt(sample.annotation_path, ndmin=2)
        return rows[:, :2].astype(np.float64)
    raise ValueError(f"Unknown dataset kind {sample.kind!r}")


# ---------------------------------------------------------------------------
# Image sizing
# ---------------------------------------------------------------------------

def resize_factor(width: int, height: int) -> float:
    """Scale factor that brings an image inside [MIN_SIDE, MAX_SIDE]."""
    longest, shortest = max(width, height), min(width, height)
    if longest > MAX_SIDE:
        return MAX_SIDE / longest
    if shortest < MIN_SIDE:
        return min(MIN_SIDE / shortest, MAX_SIDE / longest)
    return 1.0


def load_image_and_points(sample: Sample) -> Tuple[Image.Image, np.ndarray]:
    """Load an RGB image and its head points, both resized by the same factor."""
    image = Image.open(sample.image_path).convert("RGB")
    points = load_points(sample)
    factor = resize_factor(*image.size)
    if factor != 1.0:
        new_size = (max(1, round(image.size[0] * factor)), max(1, round(image.size[1] * factor)))
        image = image.resize(new_size, Image.BILINEAR)
        points = points * factor
    return image, points


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _walk(root: str, max_depth: int = 4):
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        yield dirpath


def _image_files(directory: str) -> List[str]:
    return sorted(f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def _shanghaitech_split(split_dir: str, key_prefix: str) -> List[Sample]:
    images_dir = os.path.join(split_dir, "images")
    gt_dir = next((os.path.join(split_dir, d) for d in ("ground_truth", "ground-truth")
                   if os.path.isdir(os.path.join(split_dir, d))), None)
    if gt_dir is None:
        raise FileNotFoundError(f"No ground_truth directory under {split_dir}")
    samples = []
    for name in _image_files(images_dir):
        stem = os.path.splitext(name)[0]
        ann = os.path.join(gt_dir, f"GT_{stem}.mat")
        if os.path.exists(ann):
            samples.append(Sample(os.path.join(images_dir, name), ann, "shanghaitech",
                                  f"{key_prefix}/{stem}"))
    return samples


def _qnrf_split(split_dir: str, key_prefix: str) -> List[Sample]:
    samples = []
    for name in _image_files(split_dir):
        stem = os.path.splitext(name)[0]
        ann = os.path.join(split_dir, f"{stem}_ann.mat")
        if os.path.exists(ann):
            samples.append(Sample(os.path.join(split_dir, name), ann, "qnrf", f"{key_prefix}/{stem}"))
    return samples


def _jhu_split(split_dir: str, key_prefix: str) -> List[Sample]:
    images_dir, gt_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "gt")
    samples = []
    for name in _image_files(images_dir):
        stem = os.path.splitext(name)[0]
        ann = os.path.join(gt_dir, f"{stem}.txt")
        if os.path.exists(ann):
            samples.append(Sample(os.path.join(images_dir, name), ann, "jhu", f"{key_prefix}/{stem}"))
    return samples


def find_dataset_root(data_root: str, name: str) -> Optional[str]:
    """Locate a dataset's top-level folder anywhere shallowly under ``data_root``."""
    for path in _walk(data_root):
        base = os.path.basename(path)
        if name == "sha" and base == "part_A_final":
            return path
        if name == "shb" and base == "part_B_final":
            return path
        if name == "qnrf" and os.path.isdir(os.path.join(path, "Train")) \
                and os.path.isdir(os.path.join(path, "Test")) \
                and any(f.endswith("_ann.mat") for f in os.listdir(os.path.join(path, "Train"))):
            return path
        if name == "jhu" and os.path.isdir(os.path.join(path, "train", "gt")) \
                and os.path.isdir(os.path.join(path, "test", "gt")):
            return path
    return None


def dataset_splits(data_root: str, name: str) -> Dict[str, List[Sample]]:
    """Return the dataset's official splits as {"train": [...], "test": [...], "val"?: [...]}."""
    root = find_dataset_root(data_root, name)
    if root is None:
        raise FileNotFoundError(f"Could not find {DATASET_LABELS[name]} under {data_root}")
    if name in ("sha", "shb"):
        return {split: _shanghaitech_split(os.path.join(root, f"{split}_data"), f"{name}/{split}")
                for split in ("train", "test")}
    if name == "qnrf":
        return {"train": _qnrf_split(os.path.join(root, "Train"), "qnrf/train"),
                "test": _qnrf_split(os.path.join(root, "Test"), "qnrf/test")}
    return {split: _jhu_split(os.path.join(root, split), f"jhu/{split}")
            for split in ("train", "val", "test")}


def samples_for_split_dir(split_dir: str) -> Tuple[str, List[Sample]]:
    """Identify the layout of a single split folder and index it.

    Used by the evaluation script, which is pointed directly at a test folder
    such as ``part_A_final/test_data``, ``UCF-QNRF_ECCV18/Test`` or
    ``jhu_crowd_v2.0/test``. Returns ``(kind, samples)``.
    """
    split_dir = os.path.abspath(split_dir)
    tag = os.path.basename(os.path.dirname(split_dir)) + "/" + os.path.basename(split_dir)
    has_images = os.path.isdir(os.path.join(split_dir, "images"))
    if has_images and any(os.path.isdir(os.path.join(split_dir, d)) for d in ("ground_truth", "ground-truth")):
        return "shanghaitech", _shanghaitech_split(split_dir, tag)
    if has_images and os.path.isdir(os.path.join(split_dir, "gt")):
        return "jhu", _jhu_split(split_dir, tag)
    if os.path.isdir(split_dir) and any(f.endswith("_ann.mat") for f in os.listdir(split_dir)):
        return "qnrf", _qnrf_split(split_dir, tag)
    raise FileNotFoundError(
        f"{split_dir} does not look like a ShanghaiTech, UCF-QNRF or JHU-CROWD++ split folder")


def holdout(samples: List[Sample], fraction: float, seed: int = 1337) -> Tuple[List[Sample], List[Sample]]:
    """Deterministically split ``samples`` into (kept, held_out)."""
    if fraction <= 0 or len(samples) < 2:
        return list(samples), []
    order = np.random.default_rng(seed).permutation(len(samples))
    n_hold = max(1, int(round(len(samples) * fraction)))
    held = set(order[:n_hold].tolist())
    return ([s for i, s in enumerate(samples) if i not in held],
            [s for i, s in enumerate(samples) if i in held])
