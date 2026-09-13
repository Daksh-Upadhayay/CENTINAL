"""Run the CSRNet stage over a video and collect per-frame density features."""

from typing import List, Optional

import cv2
import numpy as np
import torch

from .pipeline import DensityFeatures, FeatureExtractor, preprocess


def extract_video_features(
    video_path: str,
    model,
    device: torch.device,
    preprocess_mode: str = "raw",
    max_frames: Optional[int] = None,
    stride: int = 1,
    progress_every: int = 25,
    verbose: bool = True,
) -> np.ndarray:
    """Return an (N, 4) array of density features, one row per processed frame.

    Columns are ``count, avg_density, delta_density, spatial_variance`` -- the
    order the LSTM's scaler expects.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    extractor = FeatureExtractor()
    rows: List[DensityFeatures] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                density_map = model(preprocess(rgb, device, preprocess_mode))
            density = density_map.squeeze().cpu().numpy()
            rows.append(extractor.extract(density))

            if verbose and progress_every and len(rows) % progress_every == 0:
                print(f"  {len(rows)} frames processed", flush=True)
            if max_frames is not None and len(rows) >= max_frames:
                break
    finally:
        cap.release()

    if not rows:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack([r.as_array() for r in rows])
