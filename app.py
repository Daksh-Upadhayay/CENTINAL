import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import joblib
from scipy.ndimage import gaussian_filter
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- CONFIG ----------------
YOLO_PATH = os.path.join(BASE_DIR, "yolo_final_dense.pt")
LSTM_PATH = os.path.join(BASE_DIR, "lstm", "risk_lstm.h5")
SCALER_PATH = os.path.join(BASE_DIR, "lstm", "scaler.save")
SEQ_LEN = 30
# ----------------------------------------

st.set_page_config(page_title="Crowd Safety AI", layout="wide")
st.title("🚨 Crowd Density & Stampede Risk Monitoring System")

# Load models
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_PATH)
    lstm = tf.keras.models.load_model(LSTM_PATH)
    scaler = joblib.load(SCALER_PATH)
    return yolo, lstm, scaler

yolo, lstm, scaler = load_models()

# UI
video_path = st.text_input(
    "🎥 Video Path",
    os.path.join(BASE_DIR, "videos", "crowd_test.mp4")
)

start = st.button("▶️ Start Monitoring")

frame_col, stats_col = st.columns([2, 1])
frame_window = frame_col.empty()
heatmap_window = frame_col.empty()

count_box = stats_col.metric("👥 People Count", "0")
risk_box = stats_col.empty()

sequence_buffer = deque(maxlen=SEQ_LEN)
prev_density = 0

# ---------------- RUN ----------------
if start:
    cap = cv2.VideoCapture(video_path)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, conf=0.25)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        count = len(boxes)

        density = count / (W * H)
        delta_density = density - prev_density
        prev_density = density

        # Heatmap
        heatmap = np.zeros((H, W))
        for x1, y1, x2, y2 in boxes:
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            heatmap[cy-10:cy+10, cx-10:cx+10] += 1

        heatmap = gaussian_filter(heatmap, sigma=15)
        heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8) if heatmap.max() > 0 else heatmap.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        # Feature vector
        feature_vec = np.array([[count, density, delta_density, 0]])
        sequence_buffer.append(feature_vec[0])

        # Risk prediction
        risk_label = "SAFE"
        color = "green"

        if len(sequence_buffer) == SEQ_LEN:
            seq = np.array(sequence_buffer)
            seq_scaled = scaler.transform(seq).reshape(1, SEQ_LEN, 4)
            pred = lstm.predict(seq_scaled, verbose=0)
            risk_idx = np.argmax(pred)

            if risk_idx == 1:
                risk_label = "WARNING"
                color = "orange"
            elif risk_idx == 2:
                risk_label = "CRITICAL"
                color = "red"

        # UI updates
        count_box.metric("👥 People Count", count)
        risk_box.markdown(
            f"<h2 style='color:{color}'>⚠️ Risk Level: {risk_label}</h2>",
            unsafe_allow_html=True
        )

        frame_window.image(frame, channels="BGR", caption="Live Feed")
        heatmap_window.image(overlay, channels="BGR", caption="Density Heatmap")

    cap.release()
