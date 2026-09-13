"""Heatmap rendering for the density overlay."""

from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Normalising by the raw maximum lets a single hot pixel flatten the whole map,
# so scale against a high percentile instead and clip above it.
HEATMAP_PERCENTILE = 95
HEATMAP_SMOOTHING_SIGMA = 10

HEATMAP_OPACITY = {
    "SPARSE": 0.15,
    "MEDIUM": 0.35,
    "DENSE": 0.45,
}


def normalize_heatmap(
    density: np.ndarray,
    scene_type: str,
    frame_shape: Tuple[int, int],
) -> Tuple[np.ndarray, float]:
    """Resize, smooth and percentile-normalise a density map for display.

    Returns the 8-bit map plus the blend opacity appropriate to the scene.
    """
    h, w = frame_shape
    resized = cv2.resize(density, (w, h))
    smoothed = gaussian_filter(resized, sigma=HEATMAP_SMOOTHING_SIGMA)

    percentile_value = np.percentile(smoothed, HEATMAP_PERCENTILE)
    if percentile_value > 0:
        normalized = np.clip(smoothed / percentile_value, 0, 1)
    else:
        normalized = np.zeros_like(smoothed)

    opacity = HEATMAP_OPACITY.get(scene_type, HEATMAP_OPACITY["MEDIUM"])
    return np.uint8(255 * normalized), opacity


def overlay_heatmap(frame_bgr: np.ndarray, heatmap_8bit: np.ndarray, opacity: float) -> np.ndarray:
    """Blend a colour-mapped density heatmap over the source frame."""
    colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - opacity, colored, opacity, 0)
