# 🛡️ ML-Based Network Intrusion Detection System (NIDS)

A production-ready **Network Intrusion Detection System** built with **PyTorch**
and served as a live **Flask** web app. It classifies network traffic into five
classes using the **NSL-KDD** dataset (41 features):

| Class | Meaning |
|-------|---------|
| 🟢 **Normal** | Benign traffic |
| 🔴 **DOS**    | Denial of Service (e.g. neptune, smurf) |
| 🟡 **PROBE**  | Scanning / reconnaissance (e.g. satan, portsweep) |
| 🟠 **R2L**    | Remote-to-Local (e.g. guess_passwd, warezmaster) |
| 🟣 **U2R**    | User-to-Root privilege escalation (e.g. buffer_overflow) |

The detector uses a **two-stage (layered) strategy**:

1. A **binary** neural network decides *Normal vs Attack* (high-recall gate).
2. If flagged as an attack, a **multi-class** neural network identifies the
   specific attack type.

---

## 📐 Architecture

```
┌──────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│  Web form /  │──▶ │  Flask app   │──▶ │  Preprocessor      │──▶ │  PyTorch     │
│  JSON client │    │  (app/app.py)│    │  (scaler+encoder)  │    │  models      │
└──────────────┘    └──────────────┘    └───────────────────┘    └──────┬───────┘
        ▲                                                                │
        │                     ┌──────────────────────────┐              │
        └──────────────────── │  Normal / DOS / PROBE /   │ ◀────────────┘
                              │  R2L / U2R  + confidence  │  (binary gate → multiclass)
                              └──────────────────────────┘
```

The **exact same** preprocessing objects (`StandardScaler` + `OneHotEncoder`)
fitted during training are serialized and reloaded at inference time, so the
transformation is guaranteed identical between training and serving.

---

## 📁 Project structure

```
.
├── config.yaml               # Hyperparameters & artifact paths
├── requirements.txt
├── run_all.py                # download → preprocess → train → evaluate
├── Dockerfile / docker-compose.yml
│
├── data/
│   └── download_data.py      # Fetch NSL-KDD from public mirrors
│
├── src/                      # PyTorch backend (training + inference)
│   ├── schema.py             # 41-feature layout + attack→class mapping
│   ├── config.py             # config.yaml loader
│   ├── preprocess.py         # encode + scale, save artifacts
│   ├── dataset.py            # torch Dataset
│   ├── model.py              # BinaryClassifier / MultiClassClassifier
│   ├── train.py              # training loop (early stopping, class weights)
│   ├── evaluate.py           # test-set metrics + confusion matrix
│   └── predict.py            # two-stage inference (Predictor)
│
├── app/                      # Flask web layer
│   ├── app.py                # routes: / /predict /api/predict /health
│   ├── preprocessor.py       # form field groups + parsing
│   └── predictor.py          # artifact availability + Predictor accessor
│
├── templates/                # index.html, result.html (Jinja2)
├── static/css/style.css      # styling
├── static/js/main.js         # attack-type presets for the form
├── notebooks/EDA.ipynb       # exploratory data analysis
├── tests/test_api.py         # endpoint tests
│
├── models/                   # (generated) *.pt, *.pkl, metadata.json
└── reports/                  # (generated) metrics.json, confusion_matrix.png
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

> For a smaller CPU-only PyTorch:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. Get the dataset

```bash
python data/download_data.py
```

This downloads `KDDTrain+.txt` and `KDDTest+.txt` into `data/`. If the mirrors
are unreachable, download them manually (UNB NSL-KDD or any GitHub mirror) and
drop the two `.txt` files into `data/`.

### 3. Train the models

```bash
python -m src.train
```

This runs preprocessing automatically (if needed), trains both networks, and
writes artifacts to `models/`:
`binary_model.pt`, `multiclass_model.pt`, `scaler.pkl`, `encoder.pkl`,
`label_encoder.pkl`, `metadata.json`.

### 4. Evaluate (optional)

```bash
python -m src.evaluate
```

Prints per-class precision/recall/F1, the confusion matrix and the binary
detection rate / false-alarm rate, and saves `reports/metrics.json` +
`reports/confusion_matrix.png`.

### 5. Run the web app

```bash
python app/app.py
# open http://localhost:5000
```

### ⚡ Or do everything at once

```bash
python run_all.py      # download → preprocess → train → evaluate
python app/app.py
```

---

## 🖥️ Using the web UI

- Open `http://localhost:5000`.
- Click a **sample preset** (🟢 Normal, 🔴 DOS, 🟡 PROBE, 🟠 R2L, 🟣 U2R) to
  auto-fill a representative connection, or fill the 41 fields manually.
- Click **Analyze Traffic** to get the predicted class, a confidence bar and the
  full per-class probability breakdown.

## 🔌 JSON API

`POST /api/predict` — send any subset of the 41 features; omitted fields fall
back to benign defaults.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"protocol_type":"tcp","service":"private","flag":"S0",
       "src_bytes":0,"dst_bytes":0,"count":123,"srv_count":6,
       "serror_rate":1.0,"srv_serror_rate":1.0,"dst_host_serror_rate":1.0}}'
```

Response:

```json
{
  "predicted_class": "DOS",
  "is_attack": true,
  "confidence": 0.99,
  "attack_probability": 0.99,
  "class_probabilities": {"Normal": 0.0, "DOS": 0.99, "PROBE": 0.0, "R2L": 0.0, "U2R": 0.0}
}
```

Other routes: `GET /` (form), `POST /predict` (form submission), `GET /health`
(liveness + whether models are loaded).

---

## 🧠 Model details

**Binary classifier** — `41→one-hot ≈120 → 128 → 64 → 32 → 1`, ReLU + dropout,
`BCEWithLogitsLoss` with `pos_weight` to protect attack recall.

**Multi-class classifier** — `... → 256 (BatchNorm) → 128 → 64 → 5`, ReLU +
dropout, `CrossEntropyLoss` weighted by inverse class frequency to handle the
severe R2L/U2R imbalance.

**Training** — Adam (`lr=1e-3`, weight decay `1e-4`), `StepLR` decay, early
stopping on validation loss, stratified 85/15 train/val split of KDDTrain+.
All knobs live in [`config.yaml`](config.yaml).

---

## 📊 Results

Trained on CPU with the default [`config.yaml`](config.yaml). Evaluated on two
regimes (regenerate anytime with `python -m src.evaluate` → `reports/metrics.json`):

### In-distribution — 15% held-out validation of KDDTrain+ *(the 98%+ target regime)*

| Binary (Normal vs Attack) | Value |
|---------------------------|-------|
| Accuracy                  | **99.6%** |
| Detection rate (recall)   | **99.7%** |
| False-alarm rate          | **0.4%**  |
| ROC-AUC                   | 0.9999 |

Multi-class accuracy: **99.6%** (per-class F1 ≈ 1.00 for Normal/DOS/PROBE).

### KDDTest+ — official test set, contains novel attacks absent from training

| Binary (Normal vs Attack) | Value |
|---------------------------|-------|
| Accuracy                  | 79.9% |
| Detection rate (recall)   | 67.6% |
| Precision                 | 95.8% |
| False-alarm rate          | 4.0%  |

Multi-class accuracy: 78.2%.

> **Why the gap?** NSL-KDD's test set is deliberately constructed with attack
> variants (especially R2L/U2R) that never appear in training, so it measures
> generalisation to *unseen* attacks. ~78–80% on KDDTest+ is consistent with
> published results for this class of model — the 98%+ figure is the
> in-distribution regime. Lower `inference.attack_threshold` in `config.yaml` to
> trade a higher false-alarm rate for more detection recall.

![Confusion matrix](reports/confusion_matrix.png)

## 🐳 Docker

```bash
docker build -t nids-pytorch .
docker run -p 5000:5000 nids-pytorch
# or:
docker compose up --build
```

The image serves the app with **gunicorn** (`4 workers`) and includes a
`/health` healthcheck. Train the models first — the compose file mounts
`./models` into the container so you don't have to rebuild after retraining.

## 🏭 Production notes

- Serve with gunicorn: `gunicorn -w 4 -b 0.0.0.0:5000 app.app:app`.
- Models load **once per worker** (cached singleton) for fast inference.
- `GET /health` is suitable for load-balancer / k8s probes.
- Lower `inference.attack_threshold` in `config.yaml` to trade false alarms for
  higher detection recall.

---

## 🧪 Tests

```bash
pytest -q
```

The endpoint tests pass whether or not models are trained (they assert the
graceful 503 path when artifacts are missing).

---

## ⚙️ Configuration

Everything tunable lives in [`config.yaml`](config.yaml): dataset paths, model
widths/dropouts, optimizer settings, early-stopping patience, class-weighting
toggle and the inference threshold. Edit it and re-run `python -m src.train`.

---

## ⚠️ Disclaimer

Trained on **NSL-KDD**, an academic benchmark. Excellent for learning,
demonstrations and research, but not a drop-in production sensor for modern
live networks — real deployments need current traffic, online feature
extraction and continuous retraining. Use for education and research.
