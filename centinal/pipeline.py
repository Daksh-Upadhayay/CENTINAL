"""Frame preprocessing, density features, and scene-aware risk inference.

This module is the single source of truth for how a frame becomes a risk
decision. The Streamlit app and all evaluation scripts import from here so
that measured metrics describe the same pipeline that runs in production.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# A CSRNet checkpoint is only valid under the input scaling it was trained with,
# so the mode travels with the weights rather than being assumed at the call
# site. The original csrnet_shanghai.pth was trained on un-normalised [0, 1]
# tensors ("raw"); measured on ShanghaiTech it scores MAE 136.6 / 14.3 on
# parts A / B that way versus 181.6 / 52.4 under ImageNet statistics. The
# fine-tuned csrnet_centinal.pth continues from it and is also "raw". Checkpoints
# saved by train_csrnet.py record whichever mode they were trained with.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS_MODES = ("raw", "imagenet")
DEFAULT_PREPROCESS = "raw"

_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def preprocess(frame_rgb: np.ndarray, device: torch.device,
               mode: str = DEFAULT_PREPROCESS) -> torch.Tensor:
    """Turn an HxWx3 uint8 RGB array into an NCHW batch of one.

    ``mode`` must match the checkpoint being used; see PREPROCESS_MODES.
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocessing mode {mode!r}; expected one of {PREPROCESS_MODES}")
    tensor = _TO_TENSOR(frame_rgb)
    if mode == "imagenet":
        tensor = _NORMALIZE(tensor)
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Scene classification
# ---------------------------------------------------------------------------

SPARSE_THRESHOLD = 80    # count < 80            -> SPARSE
MEDIUM_THRESHOLD = 250   # 80 <= count < 250     -> MEDIUM
                         # count >= 250          -> DENSE


def classify_scene(count: float) -> str:
    """Bucket an estimated crowd count into a scene regime."""
    if count < SPARSE_THRESHOLD:
        return "SPARSE"
    if count < MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "DENSE"


# ---------------------------------------------------------------------------
# Density features
# ---------------------------------------------------------------------------

@dataclass
class DensityFeatures:
    """The four per-frame features the LSTM consumes, in training order."""

    count: float
    avg_density: float
    delta_density: float
    spatial_variance: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.count, self.avg_density, self.delta_density, self.spatial_variance],
            dtype=np.float64,
        )


class FeatureExtractor:
    """Converts CSRNet density maps into the LSTM's feature vector.

    Holds the previous frame's mean density so the temporal delta can be
    computed. On the first frame the delta is zero rather than the mean itself
    -- seeding the previous value at 0.0 made the first delta a spurious jump
    the size of the whole density signal.
    """

    def __init__(self) -> None:
        self._prev_avg: Optional[float] = None

    def reset(self) -> None:
        self._prev_avg = None

    def extract(self, density: np.ndarray) -> DensityFeatures:
        count = float(density.sum())
        avg = float(density.mean())
        delta = 0.0 if self._prev_avg is None else avg - self._prev_avg
        self._prev_avg = avg
        return DensityFeatures(count, avg, delta, float(density.var()))


# ---------------------------------------------------------------------------
# Risk inference
# ---------------------------------------------------------------------------

LSTM_CONFIDENCE_THRESHOLD = 0.6

RISK_LABELS = ("SAFE", "WARNING", "CRITICAL")
RISK_CSS = ("status-safe", "status-warning", "status-critical")


class RiskEngine:
    """Scene-aware hybrid risk inference over a sliding window of features.

    The window length is taken from the trained LSTM so it always matches the
    network's training horizon.
    """

    def __init__(self, lstm, scaler, seq_len: int):
        self.lstm = lstm
        self.scaler = scaler
        self.seq_len = seq_len
        self.buffer: Deque[np.ndarray] = deque(maxlen=seq_len)

    def reset(self) -> None:
        self.buffer.clear()

    @property
    def ready(self) -> bool:
        """True once enough frames have accumulated to fill the LSTM window."""
        return len(self.buffer) == self.seq_len

    def push(self, features: DensityFeatures) -> None:
        self.buffer.append(features.as_array())

    def predict(self) -> Tuple[int, float]:
        """Run the LSTM over the current window. Returns ``(class_idx, confidence)``.

        Uses a direct model call rather than ``.predict()``: Keras' predict path
        rebuilds a batched execution loop on every invocation, which dominated
        per-frame latency when called once per frame.
        """
        if not self.ready:
            return 0, 0.0
        window = np.array(self.buffer)
        scaled = self.scaler.transform(window).reshape(1, self.seq_len, -1)
        probs = self.lstm(scaled, training=False).numpy()[0]
        return int(np.argmax(probs)), float(np.max(probs))

    def active_model(self, scene_type: str, confidence: float) -> str:
        """Describe which components are driving the current decision."""
        if scene_type == "SPARSE":
            return "Count-Based Logic"
        if scene_type == "MEDIUM":
            return "CSRNet + LSTM" if confidence > LSTM_CONFIDENCE_THRESHOLD else "CSRNet Only"
        return "CSRNet + LSTM (Full)"


def compute_risk(
    scene_type: str,
    lstm_prediction_idx: int,
    lstm_confidence: float,
    has_valid_sequence: bool,
) -> int:
    """Map scene regime and LSTM output onto a risk class index.

    Returns 0 (SAFE), 1 (WARNING) or 2 (CRITICAL). The scene gate keeps the
    model from raising a stampede alarm on a crowd too small to stampede, and
    caps MEDIUM scenes at WARNING so CRITICAL requires genuine density.
    """
    if scene_type == "SPARSE":
        return 0

    if scene_type == "MEDIUM":
        if not has_valid_sequence:
            return 0
        if lstm_confidence > LSTM_CONFIDENCE_THRESHOLD:
            return 1 if lstm_prediction_idx >= 1 else 0
        # Below the confidence bar the LSTM is not trusted, but a medium-density
        # crowd still warrants an advisory rather than an all-clear.
        return 1

    if scene_type == "DENSE":
        if not has_valid_sequence:
            return 1
        if lstm_prediction_idx == 2 and lstm_confidence > LSTM_CONFIDENCE_THRESHOLD:
            return 2
        return 1 if lstm_prediction_idx >= 1 else 0

    return 0
