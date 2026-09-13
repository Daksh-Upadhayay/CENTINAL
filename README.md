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
moderate street scenes (mean 123).

| Split | Images | MAE ↓ | RMSE ↓ | MAPE | GAME(1) | GAME(2) |
|---|---|---|---|---|---|---|
| Part A (dense) | 182 | **136.60** | 226.24 | 30.1% | 144.97 | 157.12 |
| Part B (sparse/medium) | 316 | **14.28** | 21.04 | 12.1% | 17.13 | 22.35 |
| *CSRNet paper, Part A* | *182* | *68.2* | *115.0* | — | — | — |
| *CSRNet paper, Part B* | *316* | *10.6* | *16.0* | — | — | — |

MAE and RMSE are counting errors in people. MAPE is the mean per-image
percentage error. **GAME(L)** is Grid Average Mean Error: the frame is split
into a 4^L grid and per-cell absolute errors are summed, so a prediction that
gets the total right by putting people in the wrong places is still penalised.
GAME rising above MAE (145 vs 137 on Part A) is the expected signature of some
spatial smearing in the density map.

**Read this before trusting the counting stage.** The bundled checkpoint is
*not* uniformly weak — it is weak in one specific regime:

- On **Part B** it is close to the published CSRNet result (14.3 against 10.6).
- On **Part A** it is roughly **2× worse** than published (136.6 against 68.2).

In other words it behaves like a model tuned for sparse and moderate crowds.
That is the opposite of where CENTINAL needs it: the scene gate only permits a
CRITICAL alarm at counts ≥ 250, which is exactly the density range where the
counting error is largest. A mean absolute error of ~137 people on a frame
holding ~434 is not a sound basis for an automated evacuation decision, and the
system should not be deployed as one.

#### Input scaling is a property of the checkpoint

CSRNet's frontend is an ImageNet-pretrained VGG16, which would normally imply
ImageNet-normalised input. For this checkpoint that is measurably wrong:

| Split | Raw `ToTensor` (`[0,1]`) | ImageNet-normalised |
|---|---|---|
| Part A MAE | **136.60** | 181.63 |
| Part B MAE | **14.28** | 52.42 |

The bundled weights were trained on un-normalised tensors, so that is what they
must be fed. Because guessing wrong costs up to 3.7× the error, the scaling mode
is now stored *inside* the checkpoint and read back by `load_csrnet()` rather
than assumed at each call site. Reproduce the comparison with
`--preprocess raw|imagenet`.

### 2. Risk classification (LSTM)

Scored on **held-out scenario tracks** — 4,758 windows generated from tracks the
network never saw during training. Splitting is done per track, not per window:
consecutive 30-frame windows overlap heavily, so a window-level split would put
near-duplicates on both sides and inflate these numbers.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SAFE | 0.998 | 0.993 | 0.996 | 1,518 |
| WARNING | 0.994 | 0.987 | 0.991 | 1,827 |
| CRITICAL | 0.985 | **1.000** | 0.993 | 1,413 |
| **Macro avg** | **0.993** | **0.993** | **0.993** | 4,758 |

Accuracy **0.9929**, macro F1 **0.9930**. Classes are close to balanced
(32/38/30), so accuracy and macro F1 agree. Confusion matrix, rows = truth:

```
             SAFE  WARN  CRIT
SAFE         1508    10     0
WARNING         3  1803    21
CRITICAL        0     0  1413
```

No CRITICAL window is missed (recall 1.000), and no SAFE window is ever called
CRITICAL — the two failure modes that matter most in a safety system. Errors sit
on the boundary between adjacent classes, which is where a thresholded rule is
inherently ambiguous. Confidence is informative rather than flat: **0.990 on
correct predictions versus 0.824 on incorrect ones**, so the confidence gate in
`compute_risk()` is doing real work.

> **What these numbers do and do not mean.** There is no annotated stampede
> dataset here. Labels come from an explicit rule over the density signal —
> crowd level, rate of change, and turbulence — defined in `label_window()` in
> [`train_lstm.py`](train_lstm.py). So this table says the network **reproduces
> that rule on unseen scenarios with 99.3% macro F1**. It does *not* say the
> system predicts real stampedes with 99.3% accuracy, and no claim of that kind
> is supported by anything in this repository. The rule reads only observable
> density features and never the model's own output, so the evaluation is at
> least not circular — an earlier version of this evaluation derived its
> "ground truth" from the LSTM's own prediction and could only ever report
> agreement with itself.
>
> To score against real labels, use `--mode manual` with your own frame-level
> annotations.

### 3. Runtime

Measured on Apple M-series (MPS), 1920x1080 source, 60 timed frames after 5
warm-up frames, using `eval/eval_runtime.py`.

| Stage | mean ms | p50 | p95 | % of frame |
|---|---|---|---|---|
| Frame prep | 7.00 | 5.35 | 15.47 | 0.7% |
| **CSRNet inference** | **793.98** | 708.01 | 1220.38 | **81.2%** |
| Feature extraction | 0.50 | 0.38 | 0.86 | 0.1% |
| LSTM inference | 88.10 | 123.15 | 166.28 | 9.0% |
| Heatmap generation | 88.69 | 81.69 | 112.43 | 9.1% |
| **Total** | **978.32** | 910.95 | 1476.95 | |

**Throughput: 1.06 FPS** (0.64 min, 1.30 max). Peak Python heap 48.2 MB.

For comparison, the same pipeline previously measured **0.13 FPS** with CSRNet
at 6,916 ms per frame, because device selection tested only for CUDA and so fell
back to CPU on Apple hardware. Selecting MPS accounts for almost all of the
difference; outputs are identical to CPU within floating-point noise (max
relative difference 1.7e-7).

**This is not real time, and the README should not claim otherwise.** At 1 FPS
against 24 FPS source footage, CENTINAL processes roughly one frame in
twenty-four. The `--frame stride` control in the dashboard exists to make that
explicit rather than silently accumulating lag. CSRNet dominates at 81% of the
frame budget, so it is the only stage worth optimising; the remaining stages
together account for under 20%.

Note that the LSTM stage is 88 ms here only because it runs on every frame. It
was previously far worse -- the profiler recorded a 548 ms mean and 2,739 ms
maximum -- because the pipeline called `model.predict()` per frame, which
rebuilds Keras' batched execution loop on each call. A direct `model(x,
training=False)` call removed that overhead.

### 4. Fine-tuning the density estimator

`train_csrnet.py` fine-tunes CSRNet on Part A and Part B together. A time-boxed
run (6 epochs, Adam at 1e-5, ~32 minutes on MPS) produced this:

| Epoch | Part A MAE | Part B MAE | Normalised score ↓ |
|---|---|---|---|
| 0 (bundled checkpoint) | 136.66 | 14.09 | 0.214 |
| 1 | 103.42 | 46.72 | 0.308 |
| **2** | **94.73** | 21.91 | **0.198** |
| 3 | 130.80 | 37.58 | 0.302 |
| 4 | 94.96 | 29.48 | 0.228 |
| 5 | 133.08 | 19.96 | 0.234 |

The best epoch cuts dense-crowd error by **31%** (136.66 → 94.73) but raises
sparse-crowd error by **55%** (14.09 → 21.91). That checkpoint is saved as
`csrnet_centinal.pth`; **`csrnet_shanghai.pth` remains the default** so the
shipped behaviour does not change without a deliberate decision. Point
`load_csrnet()` at the other file to switch.

Two things are worth saying plainly about this result:

1. **It is a trade, not a free win.** Whether it is the right trade depends on
   which regime matters more for a given deployment. CENTINAL only permits a
   CRITICAL alarm above 250 people, which argues for the dense-accurate model;
   everyday monitoring happens in the sparse-to-medium range, which argues the
   other way.
2. **It has not converged.** Scores oscillate by more than the improvement
   between consecutive epochs, which means the run is under-trained and the
   learning rate too high for this stage. These numbers are a promising
   direction, not a finished model. A longer run at a lower learning rate is
   the obvious next step.

**Selection metric.** Checkpoints are chosen on mean *normalised* MAE -- each
split's MAE divided by its mean ground-truth count. Part A averages 434 people
per image and Part B 124, so a plain mean of the two MAEs is dominated by Part
A. Under that naive metric epoch 1 above scores "better than baseline" (75.07
against 75.38) even though it nearly quadrupled Part B error. Normalising
correctly identifies it as a regression.

---

## Known limitations

These are measured or structural, not hypothetical.

1. **Counting accuracy collapses on dense crowds.** MAE ~137 people on Part A.
   The CRITICAL alarm is gated on counts ≥ 250, so the alarm depends on the
   least reliable part of the count range. This is the single biggest obstacle
   to the system being trustworthy.
2. **Risk labels are rule-derived.** The LSTM reproduces a hand-written rule,
   it has never seen a real stampede, and its reported scores inherit every
   assumption in that rule.
3. **The scene thresholds are absolute counts**, not densities per unit area.
   They are tied to a particular camera placement and field of view; a wider or
   narrower shot changes what "80" and "250" mean.
4. **A WARNING in a MEDIUM scene is ambiguous by construction.** The rules emit
   WARNING both when the model confidently predicts elevated risk and when it is
   not confident enough to be trusted. Those are different situations.
5. **`yolo_final_dense.pt` is unused.** It ships in the repository but nothing
   loads it; counting is done entirely by CSRNet. Earlier documentation credited
   YOLOv8 for people counting, which never matched the code.
6. **Single-video calibration.** The LSTM's feature scaling is calibrated from
   one clip at one resolution. Deploying against a different camera should mean
   re-running `train_lstm.py`.

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
python eval/eval_csrnet.py --dataset_path /path/to/part_A_final/test_data --tag partA
python eval/eval_csrnet.py --dataset_path /path/to/part_B_final/test_data --tag partB

# Compare input scaling conventions
python eval/eval_csrnet.py --dataset_path /path/to/part_A_final/test_data \
    --tag partA --preprocess imagenet
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

Runs CSRNet over the video to calibrate feature scales, generates labelled
scenario tracks, fits the scaler **on the training split only**, trains with
class weighting, and writes `lstm/risk_lstm.h5`, `lstm/scaler.save`, the
held-out split `lstm/risk_testset.npz` and `lstm/training_meta.json`.

### Crowd counting

```bash
python train_csrnet.py --data_root /path/to/ShanghaiTech --part AB \
    --preprocess raw --init_from csrnet_shanghai.pth
```

Trains on Part A and Part B together and selects checkpoints on **mean MAE
across both parts**, so an improvement on dense crowds cannot quietly cost
accuracy on sparse ones.

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
├── centinal/
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
├── csrnet_shanghai.pth       # Density estimator weights
└── videos/crowd_test.mp4     # Sample footage
```

A note on the layout: CSRNet's architecture and the risk rules used to be copied
into `app.py` and each evaluation script separately. That is how the inference
sequence length drifted to 5 while the trained model expected 30 — nothing
connected the two. Everything now imports from `centinal/`, and the window
length is read from the trained model at load time rather than written down as
a constant.
