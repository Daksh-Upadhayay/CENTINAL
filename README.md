# CENTINAL

Crowd safety monitoring: estimate how many people are in a video frame, track how
that estimate moves over time, and raise a risk level when the combination looks
dangerous.

CENTINAL runs two models in sequence. **CSRNet** turns each frame into a crowd
density map whose integral is the estimated head count. A small **LSTM** reads a
30-frame window of density statistics and classifies the temporal pattern as
SAFE, WARNING or CRITICAL. A scene gate sits on top so the risk level is
interpreted against how crowded the frame actually is.

> **Scope.** This is a research prototype. The counting model is benchmarked
> against a public dataset and those numbers are below. The risk classifier is
> trained and scored against a **rule-derived** notion of risk, not against
> observed stampedes — see [Risk classification](#2-risk-classification-lstm)
> before reading anything into its accuracy.

---

## How it works

```
frame ──► CSRNet ──► density map ──┬──► count, mean, Δmean, variance
                                   │              │
                                   │              ▼
                                   │      30-frame window ──► LSTM ──► class + confidence
                                   │                                        │
                                   ▼                                        ▼
                              heatmap overlay              scene gate ──► SAFE / WARNING / CRITICAL
```

**The scene gate.** The LSTM's raw output is not the final answer. Its
prediction is combined with the estimated count so that a sparse frame cannot
raise a stampede alarm, and CRITICAL requires both a dense scene and a confident
CRITICAL prediction. The rules live in `compute_risk()` in
[`centinal/pipeline.py`](centinal/pipeline.py).

| Scene | Count | Behaviour |
|---|---|---|
| SPARSE | `< 80` | Always SAFE — too few people present for a crush |
| MEDIUM | `80–249` | Capped at WARNING; below the confidence bar it defaults to WARNING |
| DENSE | `≥ 250` | Full range; CRITICAL needs a confident CRITICAL prediction |

All inference logic lives in the `centinal/` package. The dashboard and every
evaluation script import it, so the numbers below describe the same code path
that runs in the app.

---

## Evaluation

Every number on this page was produced by the scripts in [`eval/`](eval/) on the
hardware named in each section. Nothing here is copied from a paper except the
clearly-labelled reference rows.

### 1. Crowd counting (CSRNet)

Benchmarked on the **ShanghaiTech Part A and Part B test splits** (182 and 316
images). Part A is dense crowds (mean 434 people per image); Part B is sparse to
moderate street scenes (mean 124).

| Model | Split | MAE ↓ | RMSE ↓ | MAPE | GAME(1) | GAME(2) |
|---|---|---|---|---|---|---|
| **`csrnet_centinal.pth`** (default) | Part A (dense) | **72.98** | **110.73** | 19.3% | 83.87 | 99.04 |
| **`csrnet_centinal.pth`** (default) | Part B (sparse/medium) | **13.38** | 22.77 | 15.0% | 20.53 | 28.17 |
| `csrnet_shanghai.pth` (original) | Part A | 136.60 | 226.24 | 30.1% | 144.97 | 157.12 |
| `csrnet_shanghai.pth` (original) | Part B | 14.28 | 21.04 | 12.1% | 17.13 | 22.35 |
| *CSRNet paper* | *Part A* | *68.2* | *115.0* | — | — | — |
| *CSRNet paper* | *Part B* | *10.6* | *16.0* | — | — | — |

MAE and RMSE are counting errors in people. MAPE is the mean per-image
percentage error. **GAME(L)** is Grid Average Mean Error: the frame is split
into a 4^L grid and per-cell absolute errors are summed, so a prediction that
gets the total right by putting people in the wrong places is still penalised.

The default model is the original checkpoint fine-tuned on both parts (see
[Fine-tuning](#4-fine-tuning-the-density-estimator)). It halves dense-crowd
error and is now close to the published CSRNet result on Part A. On Part B its
MAE is slightly better, but RMSE, MAPE and GAME are worse — the reason is
below.

#### Where the error is

Averages hide the pattern that matters. Pooling both test splits (498 images)
and grouping by the true crowd size:

| True count | Images | Original: MAE / bias | Default: MAE / bias |
|---|---|---|---|
| under 80 | 138 | 7.0 / −5.3 | 9.1 / +5.0 |
| 80–249 | 216 | 27.4 / −22.4 | 19.6 / +4.5 |
| 250–499 | 95 | 86.3 / −64.0 | 62.2 / +4.5 |
| 500–999 | 30 | 170.7 / −150.5 | 84.2 / −13.5 |
| 1,000+ | 19 | 482.4 / −347.6 | 188.6 / −114.3 |

The original model **undercounted systematically, and worse the bigger the
crowd** — the dangerous direction for a safety system, because it makes dense
crowds look calmer than they are. The default model is close to unbiased up to
about 500 people. It still undercounts the very largest crowds.

That shows up directly in the scene gate, which buckets frames by count:

| | Original | Default |
|---|---|---|
| Frames placed in the correct scene bucket | 85.7% | **90.6%** |
| Truly SPARSE frames kept SPARSE | **99.3%** | 92.0% |
| Truly MEDIUM frames kept MEDIUM | 85.2% | **93.5%** |
| Truly DENSE frames kept DENSE | 73.6% | **84.7%** |
| Dense frames pushed below 250, where CRITICAL cannot fire | 38 of 144 | **22 of 144** |

The one regression is at the sparse end: the default model slightly overcounts
near-empty scenes, so 11 of 138 truly sparse frames are promoted to MEDIUM and
can produce a WARNING. That costs false advisories rather than missed danger,
which is the cheaper error for this application.

**This is still not a basis for automated evacuation decisions.** 22 of 144
genuinely dense test images land below the CRITICAL gate, and crowds over 1,000
are undercounted by more than 100 people on average.

#### Input scaling is a property of the checkpoint

CSRNet's frontend is an ImageNet-pretrained VGG16, which would normally imply
ImageNet-normalised input. For the original checkpoint that is measurably wrong:

| Split | Raw `ToTensor` (`[0,1]`) | ImageNet-normalised |
|---|---|---|
| Part A MAE | **136.60** | 181.63 |
| Part B MAE | **14.28** | 52.42 |

It was trained on un-normalised tensors, and the fine-tuned default continues
from it with the same convention. Because guessing wrong costs up to 3.7× the
error, the scaling mode is stored *inside* each checkpoint and read back by
`load_csrnet()` rather than assumed at each call site. Reproduce the comparison
with `--preprocess raw|imagenet`.

### 2. Risk classification (LSTM)

Scored on **held-out scenario tracks** — 4,758 windows generated from tracks the
network never saw during training. Splitting is done per track, not per window:
consecutive 30-frame windows overlap heavily, so a window-level split would put
near-duplicates on both sides and inflate these numbers.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SAFE | 0.978 | 0.977 | 0.977 | 2,686 |
| WARNING | 0.942 | 0.936 | 0.939 | 1,478 |
| CRITICAL | 0.944 | 0.963 | 0.953 | 594 |
| **Macro avg** | **0.955** | **0.959** | **0.957** | 4,758 |

Accuracy **0.962**, macro F1 **0.957**. The classes are imbalanced (56 / 31 /
12%), so macro F1 is the number to watch. Confusion matrix, rows = truth:

```
             SAFE  WARN  CRIT
SAFE         2623    63     0
WARNING        60  1384    34
CRITICAL        0    22   572
```

No CRITICAL window is ever called SAFE and no SAFE window is ever called
CRITICAL. The misses that do happen are between neighbouring classes: 22 of 594
CRITICAL windows (3.7%) are downgraded to WARNING. Confidence is informative:
**0.983 on correct predictions versus 0.800 on incorrect ones**, so the
confidence gate in `compute_risk()` is doing real work.

**How "turbulence" is defined.** CSRNet's count wobbles from frame to frame
even when nothing in the scene changes — about 3% per frame on the sample video.
A crowd is labelled turbulent only when its frame-to-frame variability is 1.6×
what that counting noise alone would produce, and the noise level is measured
on the calibration video by `train_lstm.py`.

That definition replaces an earlier absolute threshold, which sat *below* the
counting noise. Every dense crowd therefore looked turbulent and was labelled
CRITICAL, so the classifier learned "dense means CRITICAL" and scored 99.3% on
held-out data because that task was trivially easy. The flaw stayed hidden
while the original counting model undercounted the sample video into the MEDIUM
band, where risk is capped at WARNING. Once the fine-tuned model counted it
correctly, an ordinary daytime crowd in Times Square was flagged CRITICAL on
every frame. The lower score above belongs to a harder and more meaningful task.

**On the sample video** the pipeline now reports WARNING on all 330 windows —
a dense but stable crowd — with mean confidence 0.993, including the 209 frames
that were never used for calibration. That shows the false alarm is gone on
calm footage from this camera. No surge footage is available, so nothing here
shows that CRITICAL fires correctly on a real surge.

> **What these numbers do and do not mean.** There is no annotated stampede
> dataset here. Labels come from an explicit rule over the density signal —
> crowd level, rate of change, and turbulence relative to counting noise —
> defined in `label_window()` in [`train_lstm.py`](train_lstm.py). So this table
> says the network **reproduces that rule on unseen scenarios with 95.7% macro
> F1**. It does *not* say the system predicts real stampedes with that accuracy,
> and no claim of that kind is supported by anything in this repository. The
> rule reads only observable density features and never the model's own output,
> so the evaluation is at least not circular — an earlier version of this
> evaluation derived its "ground truth" from the LSTM's own prediction and could
> only ever report agreement with itself.
>
> To score against real labels, use `--mode manual` with your own frame-level
> annotations.

### 3. Runtime

Measured on Apple M-series (MPS), 1920x1080 source, 60 timed frames after 5
warm-up frames, using `eval/eval_runtime.py`.

| Stage | mean ms | p50 | p95 | % of frame |
|---|---|---|---|---|
| Frame prep | 5.46 | 5.13 | 8.70 | 0.5% |
| **CSRNet inference** | **925.01** | 922.00 | 983.89 | **85.1%** |
| Feature extraction | 0.22 | 0.19 | 0.35 | 0.0% |
| LSTM inference | 76.41 | 119.72 | 141.48 | 7.0% |
| Heatmap generation | 79.72 | 78.78 | 83.48 | 7.3% |
| **Total** | **1086.84** | 1137.17 | 1202.98 | |

**Throughput: 0.93 FPS** (0.80 min, 1.09 max). Peak Python heap 48.2 MB.
Fine-tuning changed only the weights, not the architecture, so the per-frame
cost is the same as the original model's.

For comparison, the pipeline previously measured **0.13 FPS** with CSRNet at
6,916 ms per frame, because device selection tested only for CUDA and so fell
back to CPU on Apple hardware. Selecting MPS accounts for almost all of the
difference; outputs are identical to CPU within floating-point noise (max
relative difference 1.7e-7).

**This is not real time.** At under 1 FPS against 24 FPS source footage,
CENTINAL processes roughly one frame in twenty-six. The frame stride control
in the dashboard exists to make that explicit rather than silently accumulating
lag. CSRNet takes 85% of the frame budget, so it is the only stage worth
optimising.

The LSTM costs about 120 ms per frame once its 30-frame window has filled (the
mean is lower because the first frames of the run skip it). It was previously
far worse — a 548 ms mean and 2,739 ms maximum — because the pipeline called
`model.predict()` per frame, which rebuilds Keras' batched execution loop on
each call. A direct `model(x, training=False)` call removed that overhead.

### 4. Fine-tuning the density estimator

`train_csrnet.py` fine-tunes the original checkpoint on Part A and Part B
together. The default model came from a run on a Colab T4 GPU using
[`colab/train_csrnet_colab.ipynb`](colab/train_csrnet_colab.ipynb); the best
checkpoint was reached at **epoch 48**. Results are in the tables above and
were reproduced independently on a second machine with identical numbers.

An earlier six-epoch run on Apple Silicon (about six minutes per epoch) did not
converge — scores swung by more between epochs than the improvement itself. Two
safeguards came out of it and are built into the script:

- **An epoch-0 check.** Before training, the script scores the starting
  checkpoint. If that does not match its known benchmark result, the training
  setup disagrees with how the checkpoint was trained, and continuing would
  degrade the model. A density-scale mismatch did exactly that on the first
  attempt.
- **Selection on validation data, not test data.** The current default was
  chosen by an earlier version of the script that compared epochs on the
  ShanghaiTech *test* sets, which makes its reported test scores mildly
  optimistic. The script now holds out validation images for this.
- **A normalised selection metric.** Checkpoints are chosen on mean
  *normalised* MAE — each split's MAE divided by its mean ground-truth count.
  Part A averages 434 people per image and Part B 124, so a plain mean of the
  two MAEs is dominated by Part A. Under that naive metric, one epoch of the
  local run scored "better than baseline" (75.07 against 75.38) while nearly
  quadrupling Part B error.

---

## Known limitations

These are measured or structural, not hypothetical.

1. **The largest crowds are still undercounted.** Crowds over 1,000 are
   undercounted by 114 people on average, and 22 of 144 dense test images land
   below the count at which CRITICAL can fire.
2. **Risk labels are rule-derived.** The LSTM reproduces a hand-written rule,
   it has never seen a real stampede, and its reported scores inherit every
   assumption in that rule — including the 1.6× turbulence threshold, which is
   a judgement, not a measurement.
3. **Calibration assumes a calm crowd.** Turbulence is measured against the
   counting noise on the calibration video. Calibrating on footage of a genuine
   surge would teach the rule that surging is normal. The sample video is an
   ordinary busy street scene.
4. **Near-empty scenes are slightly overcounted.** About 8% of truly sparse
   frames are promoted to MEDIUM, which can produce an unnecessary WARNING.
5. **The scene thresholds are absolute counts**, not densities per unit area.
   They are tied to a particular camera placement and field of view; a wider or
   narrower shot changes what "80" and "250" mean.
6. **A WARNING in a MEDIUM scene is ambiguous by construction.** The rules emit
   WARNING both when the model confidently predicts elevated risk and when it is
   not confident enough to be trusted. Those are different situations.
7. **`yolo_final_dense.pt` is unused.** It ships in the repository but nothing
   loads it; counting is done entirely by CSRNet. Earlier documentation credited
   YOLOv8 for people counting, which never matched the code.
8. **Single-camera calibration.** Feature scaling and counting noise are
   calibrated from one clip at one resolution. Deploying against a different
   camera should mean re-running `train_lstm.py` on footage from it.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11+ with PyTorch and TensorFlow. CUDA and Apple Silicon (MPS) are both
picked up automatically; see `get_device()` in
[`centinal/models.py`](centinal/models.py).

## Running the dashboard

```bash
streamlit run app.py
```

Point the sidebar at a video file and press **Start Monitoring**. The sidebar
reports the active device, the LSTM window length and the input scaling in use.
The **frame stride** control processes every Nth frame, which is the practical
way to keep up with a live source.

---

## Reproducing the evaluation

### Crowd counting

The ShanghaiTech dataset is not bundled. Note that the mirror commonly linked
for it hosts only a README — the archive itself is behind the download link
*inside* that README. Unpacked, it gives `part_A_final/` and `part_B_final/`,
each with `train_data/` and `test_data/` containing `images/` and
`ground_truth/`.

```bash
python eval/eval_csrnet.py --dataset_path /path/to/part_A_final/test_data --tag partA_centinal
python eval/eval_csrnet.py --dataset_path /path/to/part_B_final/test_data --tag partB_centinal
python eval/eval_csrnet.py --dataset_path /path/to/UCF-QNRF_ECCV18/Test --tag qnrf
python eval/eval_csrnet.py --dataset_path /path/to/jhu_crowd_v2.0/test --tag jhu

# The original checkpoint, for comparison
python eval/eval_csrnet.py --model_path csrnet_shanghai.pth \
    --dataset_path /path/to/part_A_final/test_data --tag partA_shanghai

# Compare input scaling conventions
python eval/eval_csrnet.py --model_path csrnet_shanghai.pth \
    --dataset_path /path/to/part_A_final/test_data --tag partA_shanghai --preprocess imagenet
```

Writes per-image CSVs and a ground-truth-versus-prediction scatter plot to
`eval/results/`.

### Risk classification

```bash
# Held-out scenario split written by train_lstm.py
python eval/eval_lstm.py --mode holdout

# Against your own frame-level labels (CSV with frame_idx,label columns)
python eval/eval_lstm.py --mode manual --labels eval/labels.csv --video videos/crowd_test.mp4

# No labels: report what the pipeline actually does on a video
python eval/eval_lstm.py --mode video --video videos/crowd_test.mp4
```

### Runtime

```bash
python eval/eval_runtime.py --frames 100 --plot
python eval/eval_runtime.py --frames 100 --device cpu   # compare backends
```

---

## Training

### Risk classifier

```bash
python train_lstm.py --video videos/crowd_test.mp4 --epochs 80
```

Runs CSRNet over the first 120 frames of the video to calibrate feature scales
and measure the counting noise, generates labelled scenario tracks, fits the
scaler **on the training split only**, trains with class weighting, and writes
`lstm/risk_lstm.h5`, `lstm/scaler.save`, the held-out split
`lstm/risk_testset.npz` and `lstm/training_meta.json`. Use footage of a calm
crowd from the camera you intend to deploy on, and rerun this whenever the
counting model changes.

### Crowd counting

Training needs a GPU; a single epoch takes about six minutes on Apple Silicon.
The easiest route is the Colab notebook,
[`colab/train_csrnet_colab.ipynb`](colab/train_csrnet_colab.ipynb), which
downloads the dataset, trains on a free T4, and saves checkpoints to Google
Drive so a disconnect does not lose progress. On any CUDA machine:

```bash
python train_csrnet.py --data_root /path/to/datasets --datasets sha,shb,qnrf,jhu \
    --preprocess raw --init_from csrnet_centinal.pth
```

Supported datasets are ShanghaiTech A and B (`sha`, `shb`), UCF-QNRF (`qnrf`)
and JHU-CROWD++ (`jhu`); their folders are found automatically under
`--data_root`. Images larger than 2048 px or smaller than 512 px are resized
into that range, matching the preprocessing the JHU-CROWD++ authors ship.
JHU-CROWD++ is licensed for non-commercial use only.

Checkpoints are selected on **validation** data — JHU-CROWD++'s official
validation split, and a fixed 10% held out from the training sets of the
others — using mean normalised MAE, so an improvement on one dataset cannot
quietly cost accuracy on another. Test sets are only scored afterwards, with
`eval/eval_csrnet.py`.

Before the first epoch the script evaluates the starting checkpoint and prints
the result. That number must match the checkpoint's known benchmark score; if it
does not, the preprocessing or density-scale convention in training disagrees
with how the checkpoint was trained, and fine-tuning will degrade the model
rather than improve it. Ground-truth density maps use a geometry-adaptive
Gaussian kernel and are cached on first use.

---

## Project structure

```
CENTINAL/
├── app.py                    # Streamlit dashboard
├── train_lstm.py             # Trains the risk classifier
├── train_csrnet.py           # Fine-tunes the density estimator
├── colab/
│   └── train_csrnet_colab.ipynb  # GPU training on Google Colab
├── centinal/
│   ├── datasets.py           # Dataset discovery, annotations, image resizing
│   ├── models.py             # CSRNet architecture, device + checkpoint loading
│   ├── pipeline.py           # Preprocessing, features, risk inference
│   ├── video.py              # Per-frame feature extraction over a video
│   └── viz.py                # Density heatmap rendering
├── eval/
│   ├── eval_csrnet.py        # MAE / RMSE / MAPE / GAME on ShanghaiTech
│   ├── eval_lstm.py          # Risk classification metrics
│   ├── eval_runtime.py       # Stage latency, throughput, memory
│   └── results/              # Generated metrics, CSVs and plots
├── lstm/
│   ├── risk_lstm.h5          # Trained risk classifier
│   ├── scaler.save           # Feature scaler (fitted on training split only)
│   ├── risk_testset.npz      # Held-out evaluation split
│   └── training_meta.json    # Training configuration and label rule
├── csrnet_centinal.pth       # Density estimator, fine-tuned (default)
├── csrnet_shanghai.pth       # Density estimator, original (for comparison)
├── yolo_final_dense.pt       # Unused; see Known limitations
└── videos/crowd_test.mp4     # Sample footage
```

A note on the layout: CSRNet's architecture and the risk rules used to be copied
into `app.py` and each evaluation script separately. That is how the inference
sequence length drifted to 5 while the trained model expected 30 — nothing
connected the two. Everything now imports from `centinal/`, and the window
length is read from the trained model at load time rather than written down as
a constant.
