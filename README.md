# 🚨 CENTINAL: Crowd Safety & Stampede Risk Monitoring System

CENTINAL is an AI-powered crowd safety monitoring system designed to detect and predict stampede risks in real-time. It leverages Computer Vision (YOLO) for crowd density estimation and Deep Learning (LSTM) for risk sequence analysis.

## 🚀 Features

- **Real-time People Counting**: Uses YOLOv8 (Dense) to accurately detect and count individuals in video feeds.
- **Crowd Density Heatmap**: Generates a dynamic heatmap to visualize high-density areas.
- **Stampede Risk Prediction**: Analyzes time-series data using an LSTM model to predict risk levels:
  - 🟢 **SAFE**: Normal crowd flow.
  - 🟠 **WARNING**: High density, potential risk.
  - 🔴 **CRITICAL**: Immediate stampede risk detected.
- **Interactive Dashboard**: Built with Streamlit for easy visualization and monitoring.

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Object Detection**: ultralytics (YOLO)
- **Time-Series Analysis**: TensorFlow/Keras (LSTM)
- **Image Processing**: OpenCV, SciPy
- **Data Handling**: NumPy, Joblib

## 📂 Project Structure

```
CENTINAL/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── yolo_final_dense.pt     # Trained YOLO model object detection
├── lstm/
│   ├── risk_lstm.h5        # Trained LSTM model for risk prediction
│   └── scaler.save         # Scaler for normalizing input data
└── videos/
    └── crowd_test.mp4      # Default test video
```

## ⚙️ Installation

1.  **Clone the repository** (or navigate to the project directory):
    ```bash
    cd /path/to/CENTINAL
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2.  **Monitor**:
    - The dashboard will open in your browser.
    - Enter the path to a video file or use the default test video.
    - Click **Start Monitoring** to view real-time analysis.

## 🧠 Model Details

- **Input**: Video frames are processed to extract crowd count and density metrics.
- **Sequence Processing**: A sliding window of 30 frames (SEQ_LEN) is used as input for the LSTM.
- **Output**: The system outputs a risk classification (Safe/Warning/Critical) updated in real-time.
