"""Model definitions and loading for CENTINAL.

Both the CSRNet density estimator and the LSTM risk head are loaded here so
that the Streamlit app and every evaluation script share one implementation.
Duplicating the architecture across call sites is what previously allowed the
inference sequence length to drift away from the trained one.
"""

import os
from typing import Tuple

import joblib
import torch
import torch.nn as nn
from torchvision.models import vgg16

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSRNET_PATH = os.path.join(BASE_DIR, "csrnet_shanghai.pth")
LSTM_PATH = os.path.join(BASE_DIR, "lstm", "risk_lstm.h5")
SCALER_PATH = os.path.join(BASE_DIR, "lstm", "scaler.save")


class CSRNet(nn.Module):
    """CSRNet: VGG16 frontend + dilated-convolution backend producing a density map.

    The frontend is the first 23 layers of VGG16 (through conv4_3), so the
    density map is 1/8 the spatial resolution of the input. Summing the map
    gives the estimated crowd count.
    """

    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        return self.backend(self.frontend(x))


def get_device() -> torch.device:
    """Pick the best available torch device (CUDA > Apple MPS > CPU).

    Apple Silicon reaches MPS here; the pipeline previously checked only for
    CUDA and so ran on CPU on every Mac, which is where most of the missing
    throughput went.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # torch.backends.mps is absent on older builds and on non-Apple platforms.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_csrnet(path: str = CSRNET_PATH, device: torch.device = None):
    """Load CSRNet weights in eval mode.

    Returns ``(model, device, meta)``. ``meta["preprocess"]`` names the input
    scaling the checkpoint was trained with, so callers never have to guess.
    Checkpoints saved by ``train_csrnet.py`` embed that alongside the weights;
    a bare state_dict is assumed to be the original raw-input model.
    """
    device = device or get_device()
    blob = torch.load(path, map_location=device, weights_only=False)

    if isinstance(blob, dict) and "state_dict" in blob:
        state_dict = blob["state_dict"]
        meta = {k: v for k, v in blob.items() if k != "state_dict"}
        meta.setdefault("preprocess", "imagenet")
    else:
        state_dict = blob
        meta = {"preprocess": "raw"}

    model = CSRNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, meta


def load_risk_head(lstm_path: str = LSTM_PATH, scaler_path: str = SCALER_PATH):
    """Load the LSTM risk classifier and its feature scaler.

    Returns ``(lstm, scaler, seq_len)``. The sequence length is read off the
    loaded model rather than hardcoded, so inference can never silently run on
    a different window than the one the network was trained with.
    """
    import tensorflow as tf  # imported lazily; torch-only callers skip the cost

    lstm = tf.keras.models.load_model(lstm_path, compile=False)
    scaler = joblib.load(scaler_path)

    seq_len = lstm.input_shape[1]
    if seq_len is None:
        raise ValueError(
            f"{lstm_path} has a variable-length time axis, so the training "
            "window cannot be recovered. Retrain with a fixed input shape."
        )
    return lstm, scaler, int(seq_len)
