import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16
import tensorflow as tf
import joblib
from collections import deque
from scipy.ndimage import gaussian_filter
from typing import Tuple

# ============================================================================
# CONFIGURATION - Named thresholds for scene-aware inference
# ============================================================================

# Scene classification thresholds based on estimated crowd count
SPARSE_THRESHOLD = 80       # count < 80 → SPARSE scene
MEDIUM_THRESHOLD = 250      # 80 ≤ count < 250 → MEDIUM scene
                            # count ≥ 250 → DENSE scene

# LSTM confidence threshold for model activation in MEDIUM scenes
LSTM_CONFIDENCE_THRESHOLD = 0.6

# Heatmap normalization percentile (avoids outlier-driven max normalization)
HEATMAP_PERCENTILE = 95

# Heatmap opacity by scene type
HEATMAP_OPACITY_SPARSE = 0.15   # Very faint for sparse scenes
HEATMAP_OPACITY_MEDIUM = 0.35  # Moderate for medium density
HEATMAP_OPACITY_DENSE = 0.45   # Full visibility for dense crowds

# Gaussian smoothing sigma for heatmap
HEATMAP_SMOOTHING_SIGMA = 10

# Sequence length for LSTM temporal analysis
SEQ_LEN = 5


# ============================================================================
# SCENE CLASSIFICATION
# ============================================================================

def classify_scene(count: int) -> str:
    """
    Classify the current scene based on estimated crowd count.
    
    Scene types determine which models are active and how risk is computed:
    - SPARSE: Low crowd density, simple count-based logic, always SAFE
    - MEDIUM: Moderate density, conditional LSTM usage based on confidence
    - DENSE: High density, full CSRNet + LSTM pipeline, allows CRITICAL
    
    Args:
        count: Estimated crowd count from CSRNet density map
        
    Returns:
        Scene type string: 'SPARSE', 'MEDIUM', or 'DENSE'
    """
    if count < SPARSE_THRESHOLD:
        return "SPARSE"
    elif count < MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "DENSE"


def get_active_model(scene_type: str, lstm_confidence: float) -> str:
    """
    Determine which model(s) are actively driving inference.
    
    Args:
        scene_type: Current scene classification
        lstm_confidence: Confidence score of LSTM prediction (max softmax value)
        
    Returns:
        Description of active model configuration
    """
    if scene_type == "SPARSE":
        return "Count-Based Logic"
    elif scene_type == "MEDIUM":
        if lstm_confidence > LSTM_CONFIDENCE_THRESHOLD:
            return "CSRNet + LSTM"
        else:
            return "CSRNet Only"
    else:  # DENSE
        return "CSRNet + LSTM (Full)"


# ============================================================================
# HYBRID RISK INFERENCE
# ============================================================================

def compute_risk(
    scene_type: str,
    count: int,
    lstm_prediction_idx: int,
    lstm_confidence: float,
    has_valid_sequence: bool
) -> Tuple[str, str]:
    """
    Compute risk level using scene-aware hybrid inference logic.
    
    This function implements conservative risk prediction that prevents
    false alarms in sparse scenes while allowing appropriate escalation
    in genuinely dangerous dense crowd situations.
    
    Risk escalation rules:
    - SPARSE: Always returns SAFE (low crowd = low stampede risk)
    - MEDIUM: Returns WARNING if LSTM confidence > threshold, else WARNING default
    - DENSE: Full LSTM prediction allowed, CRITICAL only with high confidence
    
    Args:
        scene_type: Current scene classification ('SPARSE', 'MEDIUM', 'DENSE')
        count: Estimated crowd count
        lstm_prediction_idx: LSTM output class (0=SAFE, 1=WARNING, 2=CRITICAL)
        lstm_confidence: Maximum softmax probability from LSTM
        has_valid_sequence: Whether we have enough frames for LSTM inference
        
    Returns:
        Tuple of (risk_label, css_class) for UI display
    """
    # SPARSE scenes: Always safe - insufficient crowd for stampede risk
    if scene_type == "SPARSE":
        return "SAFE", "status-safe"
    
    # MEDIUM scenes: Conditional LSTM usage based on confidence
    if scene_type == "MEDIUM":
        if not has_valid_sequence:
            # Not enough temporal data yet - default to safe
            return "SAFE", "status-safe"
        
        if lstm_confidence > LSTM_CONFIDENCE_THRESHOLD:
            # Trust LSTM prediction but cap at WARNING (no CRITICAL for medium)
            if lstm_prediction_idx >= 1:
                return "WARNING", "status-warning"
            return "SAFE", "status-safe"
        else:
            # Low confidence - default to WARNING as precaution
            return "WARNING", "status-warning"
    
    # DENSE scenes: Full LSTM inference with CRITICAL allowed
    if scene_type == "DENSE":
        if not has_valid_sequence:
            # No temporal data but high density - cautionary WARNING
            return "WARNING", "status-warning"
        
        # Full LSTM prediction with confidence gating for CRITICAL
        if lstm_prediction_idx == 2 and lstm_confidence > LSTM_CONFIDENCE_THRESHOLD:
            return "CRITICAL", "status-critical"
        elif lstm_prediction_idx >= 1:
            return "WARNING", "status-warning"
        else:
            # Even SAFE from LSTM in dense crowd warrants attention
            return "SAFE", "status-safe"
    
    # Fallback (should never reach here)
    return "SAFE", "status-safe"


# ============================================================================
# HEATMAP NORMALIZATION
# ============================================================================

def normalize_heatmap(
    density: np.ndarray,
    scene_type: str,
    frame_shape: Tuple[int, int]
) -> Tuple[np.ndarray, float]:
    """
    Apply percentile-based normalization with Gaussian smoothing to density map.
    
    Uses 95th percentile normalization instead of max() to avoid outlier-driven
    scaling that can make heatmaps appear uniformly hot or cold. Applies
    scene-appropriate opacity for visual clarity.
    
    Args:
        density: Raw density map from CSRNet
        scene_type: Current scene classification for opacity selection
        frame_shape: (height, width) of original frame for resizing
        
    Returns:
        Tuple of (normalized_heatmap, opacity) for overlay blending
    """
    h, w = frame_shape
    
    # Resize density map to match frame dimensions
    density_resized = cv2.resize(density, (w, h))
    
    # Apply Gaussian smoothing for visual continuity
    density_smoothed = gaussian_filter(density_resized, sigma=HEATMAP_SMOOTHING_SIGMA)
    
    # Percentile-based normalization (robust to outliers)
    percentile_value = np.percentile(density_smoothed, HEATMAP_PERCENTILE)
    
    if percentile_value > 0:
        # Clip values above percentile threshold and normalize
        density_normalized = np.clip(density_smoothed / percentile_value, 0, 1)
    else:
        # Handle edge case of zero/near-zero density
        density_normalized = np.zeros_like(density_smoothed)
    
    # Select opacity based on scene type
    opacity_map = {
        "SPARSE": HEATMAP_OPACITY_SPARSE,
        "MEDIUM": HEATMAP_OPACITY_MEDIUM,
        "DENSE": HEATMAP_OPACITY_DENSE
    }
    opacity = opacity_map.get(scene_type, HEATMAP_OPACITY_MEDIUM)
    
    # Convert to 8-bit for colormap application
    heatmap_8bit = np.uint8(255 * density_normalized)
    
    return heatmap_8bit, opacity


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Centinal | Crowd Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
.metric-card {
    background: #161b22;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    text-align: center;
    margin-bottom: 12px;
}
.metric-title {
    font-size: 14px;
    color: #9ba3af;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
}
.decision-card {
    background: #1a1f2e;
    padding: 16px;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    margin-bottom: 12px;
}
.decision-title {
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.decision-value {
    font-size: 16px;
    font-weight: 500;
    color: #e5e7eb;
}
.scene-sparse { border-left-color: #22c55e; }
.scene-medium { border-left-color: #facc15; }
.scene-dense { border-left-color: #ef4444; }
.status-safe { color: #22c55e; }
.status-warning { color: #facc15; }
.status-critical { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSRNET_PATH = os.path.join(BASE_DIR, "csrnet_shanghai.pth")
LSTM_PATH = os.path.join(BASE_DIR, "lstm", "risk_lstm.h5")
SCALER_PATH = os.path.join(BASE_DIR, "lstm", "scaler.save")


# ============================================================================
# CSRNET ARCHITECTURE
# ============================================================================

class CSRNet(nn.Module):
    """
    CSRNet for crowd density estimation.
    Uses VGG16 frontend with dilated convolution backend for density map generation.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=False)
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
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.backend(self.frontend(x))


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load CSRNet, LSTM, and scaler with caching for performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csrnet = CSRNet().to(device)
    csrnet.load_state_dict(torch.load(CSRNET_PATH, map_location=device))
    csrnet.eval()

    lstm = tf.keras.models.load_model(LSTM_PATH)
    scaler = joblib.load(SCALER_PATH)

    return csrnet, lstm, scaler, device

csrnet, lstm, scaler, device = load_models()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("Centinal")
st.sidebar.caption("Crowd Safety Intelligence System")

video_path = st.sidebar.text_input(
    "Video Source",
    value=os.path.join(BASE_DIR, "videos", "crowd_test.mp4")
)

start_btn = st.sidebar.button("▶ Start Monitoring")

# ============================================================================
# MAIN LAYOUT
# ============================================================================

st.markdown("## 🚨 Live Crowd Risk Monitoring")

frame_col, stats_col = st.columns([3, 1])

frame_placeholder = frame_col.empty()
heatmap_placeholder = frame_col.empty()

# Metric and decision card placeholders
with stats_col:
    scene_box = st.empty()
    model_box = st.empty()
    count_box = st.empty()
    risk_box = st.empty()

# ============================================================================
# TRANSFORMS AND BUFFERS
# ============================================================================

transform = transforms.ToTensor()
sequence_buffer = deque(maxlen=SEQ_LEN)
prev_density = 0

# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

if start_btn:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Error: Could not open video at path: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame for CSRNet input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # ---- CSRNet Density Estimation ----
        with torch.no_grad():
            density_map = csrnet(input_tensor)

        density = density_map.squeeze().cpu().numpy()
        count = int(density.sum())

        # ---- Scene Classification ----
        scene_type = classify_scene(count)

        # ---- Feature Extraction for LSTM ----
        avg_density = density.mean()
        delta_density = avg_density - prev_density
        prev_density = avg_density
        spatial_variance = density.var()

        features = np.array([[count, avg_density, delta_density, spatial_variance]])
        sequence_buffer.append(features[0])

        # ---- LSTM Inference ----
        has_valid_sequence = len(sequence_buffer) == SEQ_LEN
        lstm_prediction_idx = 0
        lstm_confidence = 0.0

        if has_valid_sequence:
            seq = np.array(sequence_buffer)
            seq_scaled = scaler.transform(seq).reshape(1, SEQ_LEN, 4)
            pred = lstm.predict(seq_scaled, verbose=0)
            lstm_prediction_idx = int(np.argmax(pred))
            lstm_confidence = float(np.max(pred))

        # ---- Hybrid Risk Computation ----
        risk_label, risk_class = compute_risk(
            scene_type=scene_type,
            count=count,
            lstm_prediction_idx=lstm_prediction_idx,
            lstm_confidence=lstm_confidence,
            has_valid_sequence=has_valid_sequence
        )

        # ---- Active Model Determination ----
        active_model = get_active_model(scene_type, lstm_confidence)

        # ---- Heatmap Generation ----
        h, w = frame.shape[:2]
        heatmap_normalized, heatmap_opacity = normalize_heatmap(
            density=density,
            scene_type=scene_type,
            frame_shape=(h, w)
        )
        
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            frame, 1.0 - heatmap_opacity,
            heatmap_colored, heatmap_opacity,
            0
        )

        # ---- UI Updates ----
        frame_placeholder.image(frame, channels="BGR", caption="Live Feed")
        heatmap_placeholder.image(overlay, channels="BGR", caption="Density Heatmap")

        # Scene type indicator
        scene_class = f"scene-{scene_type.lower()}"
        with stats_col:
            scene_box.markdown(f"""
            <div class="decision-card {scene_class}">
                <div class="decision-title">Scene Type</div>
                <div class="decision-value">{scene_type}</div>
            </div>
            """, unsafe_allow_html=True)

            model_box.markdown(f"""
            <div class="decision-card">
                <div class="decision-title">Active Model</div>
                <div class="decision-value">{active_model}</div>
            </div>
            """, unsafe_allow_html=True)

            count_box.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Crowd Count</div>
                <div class="metric-value">{count}</div>
            </div>
            """, unsafe_allow_html=True)

            risk_box.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Risk Level</div>
                <div class="metric-value {risk_class}">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

    cap.release()
