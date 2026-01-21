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

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Centinal | Crowd Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
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
}
.metric-title {
    font-size: 14px;
    color: #9ba3af;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
}
.status-safe { color: #22c55e; }
.status-warning { color: #facc15; }
.status-critical { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSRNET_PATH = os.path.join(BASE_DIR, "csrnet_shanghai.pth")
LSTM_PATH = os.path.join(BASE_DIR, "lstm", "risk_lstm.h5")
SCALER_PATH = os.path.join(BASE_DIR, "lstm", "scaler.save")

SEQ_LEN = 5

# -------------------- CSRNET MODEL --------------------
class CSRNet(nn.Module):
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

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csrnet = CSRNet().to(device)
    csrnet.load_state_dict(torch.load(CSRNET_PATH, map_location=device))
    csrnet.eval()

    lstm = tf.keras.models.load_model(LSTM_PATH)
    scaler = joblib.load(SCALER_PATH)

    return csrnet, lstm, scaler, device

csrnet, lstm, scaler, device = load_models()

# -------------------- SIDEBAR --------------------
st.sidebar.title("Centinal")
st.sidebar.caption("Crowd Safety Intelligence System")

video_path = st.sidebar.text_input(
    "Video Source",
    value=os.path.join(BASE_DIR, "videos", "crowd_test.mp4")
)

start_btn = st.sidebar.button("▶ Start Monitoring")

# -------------------- HEADER --------------------
st.markdown("## 🚨 Live Crowd Risk Monitoring")

frame_col, stats_col = st.columns([3, 1])

frame_placeholder = frame_col.empty()
heatmap_placeholder = frame_col.empty()

# -------------------- METRIC CARDS --------------------
with stats_col:
    count_box = st.empty()
    risk_box = st.empty()

# -------------------- TRANSFORMS --------------------
transform = transforms.ToTensor()
sequence_buffer = deque(maxlen=SEQ_LEN)
prev_density = 0

# -------------------- RUN --------------------
if start_btn:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Error: Could not open video at path: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            density_map = csrnet(input_tensor)

        density = density_map.squeeze().cpu().numpy()
        count = int(density.sum())

        h, w = frame.shape[:2]
        density_resized = cv2.resize(density, (w, h))
        density_resized = gaussian_filter(density_resized, sigma=10)

        avg_density = density.mean()
        delta_density = avg_density - prev_density
        prev_density = avg_density
        spatial_variance = density.var()

        features = np.array([[count, avg_density, delta_density, spatial_variance]])
        sequence_buffer.append(features[0])

        # ---- LSTM RISK ----
        risk_label = "SAFE"
        risk_class = "status-safe"

        if len(sequence_buffer) == SEQ_LEN:
            seq = np.array(sequence_buffer)
            seq_scaled = scaler.transform(seq).reshape(1, SEQ_LEN, 4)
            pred = lstm.predict(seq_scaled, verbose=0)
            idx = np.argmax(pred)

            if idx == 1:
                risk_label = "WARNING"
                risk_class = "status-warning"
            elif idx == 2:
                risk_label = "CRITICAL"
                risk_class = "status-critical"

        # ---- UI ----
        heatmap = cv2.applyColorMap(
            np.uint8(255 * density_resized / density_resized.max()),
            cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        frame_placeholder.image(frame, channels="BGR", caption="Live Feed")
        heatmap_placeholder.image(overlay, channels="BGR", caption="Density Heatmap")

        with stats_col:
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

