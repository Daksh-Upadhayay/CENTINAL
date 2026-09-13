"""CENTINAL - Streamlit dashboard for live crowd risk monitoring.

All model and inference logic lives in the ``centinal`` package so that this
dashboard and the evaluation scripts in ``eval/`` exercise the same pipeline.
"""

import os

import cv2
import numpy as np
import streamlit as st
import torch

from centinal.models import BASE_DIR, load_csrnet, load_risk_head
from centinal.pipeline import (
    RISK_CSS,
    RISK_LABELS,
    FeatureExtractor,
    RiskEngine,
    classify_scene,
    compute_risk,
    preprocess,
)
from centinal.viz import normalize_heatmap, overlay_heatmap

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Centinal | Crowd Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
body { background-color: #0e1117; color: #fafafa; }
.metric-card {
    background: #161b22; padding: 20px; border-radius: 14px;
    box-shadow: 0 0 20px rgba(0,0,0,0.4); text-align: center; margin-bottom: 12px;
}
.metric-title { font-size: 14px; color: #9ba3af; }
.metric-value { font-size: 28px; font-weight: 600; }
.decision-card {
    background: #1a1f2e; padding: 16px; border-radius: 12px;
    border-left: 4px solid #3b82f6; margin-bottom: 12px;
}
.decision-title {
    font-size: 12px; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px;
}
.decision-value { font-size: 16px; font-weight: 500; color: #e5e7eb; }
.scene-sparse { border-left-color: #22c55e; }
.scene-medium { border-left-color: #facc15; }
.scene-dense { border-left-color: #ef4444; }
.status-safe { color: #22c55e; }
.status-warning { color: #facc15; }
.status-critical { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load CSRNet and the LSTM risk head once per session."""
    csrnet, device, meta = load_csrnet()
    lstm, scaler, seq_len = load_risk_head()
    return csrnet, device, meta, lstm, scaler, seq_len


csrnet, device, csrnet_meta, lstm, scaler, seq_len = load_models()
preprocess_mode = csrnet_meta["preprocess"]

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("Centinal")
st.sidebar.caption("Crowd Safety Intelligence System")

video_path = st.sidebar.text_input(
    "Video Source",
    value=os.path.join(BASE_DIR, "videos", "crowd_test.mp4"),
)

frame_stride = st.sidebar.slider(
    "Frame stride", min_value=1, max_value=10, value=1,
    help="Process every Nth frame. Raise this to keep up with a live feed on slower hardware.",
)

start_btn = st.sidebar.button("▶ Start Monitoring")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Device: `{device}`  ·  LSTM window: `{seq_len}` frames  ·  input: `{preprocess_mode}`"
)

# ============================================================================
# MAIN LAYOUT
# ============================================================================

st.markdown("## 🚨 Live Crowd Risk Monitoring")

frame_col, stats_col = st.columns([3, 1])
frame_placeholder = frame_col.empty()
heatmap_placeholder = frame_col.empty()

with stats_col:
    scene_box = st.empty()
    model_box = st.empty()
    count_box = st.empty()
    risk_box = st.empty()

# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

if start_btn:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Error: Could not open video at path: {video_path}")
    else:
        features = FeatureExtractor()
        risk_engine = RiskEngine(lstm, scaler, seq_len)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb, device, preprocess_mode)

            with torch.no_grad():
                density_map = csrnet(input_tensor)
            density = density_map.squeeze().cpu().numpy()

            feats = features.extract(density)
            count = int(feats.count)
            scene_type = classify_scene(count)

            risk_engine.push(feats)
            pred_idx, confidence = risk_engine.predict()

            risk_idx = compute_risk(
                scene_type=scene_type,
                lstm_prediction_idx=pred_idx,
                lstm_confidence=confidence,
                has_valid_sequence=risk_engine.ready,
            )
            risk_label, risk_class = RISK_LABELS[risk_idx], RISK_CSS[risk_idx]
            active_model = risk_engine.active_model(scene_type, confidence)

            h, w = frame.shape[:2]
            heatmap_8bit, opacity = normalize_heatmap(density, scene_type, (h, w))
            overlay = overlay_heatmap(frame, heatmap_8bit, opacity)

            frame_placeholder.image(frame, channels="BGR", caption="Live Feed")
            heatmap_placeholder.image(overlay, channels="BGR", caption="Density Heatmap")

            warmup = "" if risk_engine.ready else f" (warming up {len(risk_engine.buffer)}/{seq_len})"

            with stats_col:
                scene_box.markdown(f"""
                <div class="decision-card scene-{scene_type.lower()}">
                    <div class="decision-title">Scene Type</div>
                    <div class="decision-value">{scene_type}</div>
                </div>
                """, unsafe_allow_html=True)

                model_box.markdown(f"""
                <div class="decision-card">
                    <div class="decision-title">Active Model</div>
                    <div class="decision-value">{active_model}{warmup}</div>
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
                    <div class="metric-title">confidence {confidence:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

        cap.release()
