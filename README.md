# 🛡️ ML-Based Network Intrusion Detection System (NIDS)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen)
![Validation](https://img.shields.io/badge/validation-99.6%25-brightgreen)
![KDDTest+](https://img.shields.io/badge/KDDTest%2B-80.0%25-yellow)

An educational **network intrusion classification demo** built with **PyTorch**
and served as a **Flask** web app. It classifies 41-feature **NSL-KDD** records
into five classes. It does not capture live network traffic or provide a
validated production security control:

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

> **Why two stages instead of one 5-class model?** The two questions carry very
> different costs. Missing an attack outright is far worse than mislabelling
> which family it belongs to, so stage 1 optimises purely for recall behind a
> single tunable dial (`inference.attack_threshold`) that trades false alarms
> for detection. Stage 2 then types the attack *already knowing* it is hostile,
> so it never spends capacity separating 67,343 normal connections from 52 U2R
> ones — it only has to tell four attack families apart.

**Contents** — [Architecture](#-architecture) · [Dataset](#-dataset) ·
[Structure](#-project-structure) · [Quickstart](#-quickstart) ·
[Web UI](#-using-the-web-ui) · [JSON API](#-json-api) ·
[Model details](#-model-details) · [Results](#-results) ·
[Production](#-production-notes) · [Troubleshooting](#-troubleshooting) ·
[Configuration](#-configuration)

For a live-traffic pilot, follow the [step-by-step implementation manual](docs/live_ids_manual.md).
For a Windows-only pilot on your own PC, start with the [Windows procedure](docs/windows_pc_pilot.md).

The separate `live_ids` package parses Zeek JSON `conn.log` records and a
minimized Windows Security 4625 export. It can replay a **provisional,
observation-only** rule: four failed logons in a rolling hour. It does not pass
those records to the NSL-KDD model or classify them as attacks. From an elevated
PowerShell window, export only timestamps, record IDs, logon types and status
codes to a folder outside Git. The [live data contract](docs/live_data_schema.md)
documents the fields and limitations:

```powershell
.\scripts\export_windows_auth.ps1
.\.venv\Scripts\python.exe -m live_ids replay --auth "$env:USERPROFILE\nids-pilot\auth_failures_7d.csv"
```

Add `--labels` for a locally reviewed `record_id,label` CSV and `--conn` plus
`--capture-id` for a Windows-accessible Zeek JSON `conn.log`. The replay prints
aggregate counts and candidate-alert metadata, not account names or IPs. Run
`.\scripts\watch_windows_auth.ps1` in an elevated PowerShell window for a
bounded 10-minute live check; it never blocks a sign-in. Pilot data and alerts
stay under `%USERPROFILE%\nids-pilot`, outside this repository. A new live ML
model and validated attack recall still require labeled attack scenarios.

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

## 🗃️ Dataset

**NSL-KDD** — the de-duplicated successor to KDD Cup '99, with the redundant
records that let naive models score artificially high removed.

| Class | Train | % | KDDTest+ | % | Note |
|-------|------:|------:|---------:|------:|------|
| 🟢 Normal | 67,343 | 53.46% | 9,711 | 43.08% | |
| 🔴 DOS    | 45,927 | 36.46% | 7,460 | 33.09% | |
| 🟡 PROBE  | 11,656 |  9.25% | 2,421 | 10.74% | |
| 🟠 R2L    |    995 |  0.79% | 2,885 | 12.80% | **16× over-represented in test** |
| 🟣 U2R    |     52 |  0.04% |    67 |  0.30% | 7× over-represented in test |
| **Total** | **125,973** | | **22,544** | | |

Two properties of this table drive everything else in the project:

1. **Extreme training imbalance** — 67,343 Normal connections against **52**
   U2R, a 1,295:1 ratio. Unweighted training simply ignores U2R and still
   scores well, so the multi-class loss is weighted by inverse class frequency
   (U2R receives ~487× the weight of Normal) to force the model to care about
   the rare classes.
2. **KDDTest+ is a deliberately different distribution.** R2L is 0.79% of the
   training set but 12.80% of the test set, and most of those test records are
   attack variants that never appear in training. This is the single reason the
   two result regimes below diverge so sharply — and why quoting only the
   validation figure would misrepresent the system.

**Feature encoding** — every connection carries 41 features: **38 numeric**
(z-score standardised) and **3 symbolic** (`protocol_type`, `service`, `flag`)
one-hot expanded into **84** columns, for a final **122-dimensional** input
vector.

---

## 📁 Project structure

```
.
├── config.yaml               # Hyperparameters & artifact paths
├── requirements.txt
├── run_all.py                # download → preprocess → train → evaluate
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

### 🔒 Securing the endpoints

The API key is optional for local demos. Set it before exposing the app on a
network. The built-in request limit defaults to 120 requests per minute per
worker; it is a local safeguard, not a replacement for a limit at the network
edge:

| Env var | Effect | Default |
|---------|--------|---------|
| `NIDS_API_KEY` | When set, the API requires `X-API-Key`. The browser form requires HTTP Basic with user `nids` and the key as password. | unset → open |
| `NIDS_RATE_LIMIT` | Max requests/minute per peer IP and worker on the form and inference routes. `0` disables. | `120` |

```bash
NIDS_API_KEY=your-secret NIDS_RATE_LIMIT=60 python app/app.py

curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" -H "X-API-Key: your-secret" \
  -d '{"features": {"protocol_type":"tcp","service":"http","flag":"SF"}}'
```

Model checkpoints are loaded with `torch.load(..., weights_only=True)`.
The `.pkl` preprocessing artifacts use `joblib` and must come from a trusted
source; loading an untrusted pickle can execute code.

Invalid JSON shapes and feature values return HTTP 400. The app ignores
client-supplied `X-Forwarded-For`; configure trusted proxy handling at your
deployment boundary if you need the original client IP. Use TLS when sending
the API key or browser Basic credentials.

---

## 🧠 Model details

**Binary classifier** — `122 → 128 → 64 → 32 → 1`, ReLU + dropout
`[0.3, 0.2, 0.0]`, `BCEWithLogitsLoss` with `pos_weight` to protect attack
recall.

**Multi-class classifier** — `122 → 256 (BatchNorm) → 128 → 64 → 5`, ReLU +
dropout `[0.4, 0.3, 0.2]`, `CrossEntropyLoss` weighted by inverse class
frequency to handle the severe R2L/U2R imbalance. BatchNorm sits on the widest
layer only, where it stabilises training against that imbalance.

**Training** — Adam (`lr=1e-3`, weight decay `1e-4`), `StepLR` decay (×0.5
every 15 epochs), early stopping on validation loss (patience 10), stratified
85/15 train/val split of KDDTrain+, batch size 256, seed 42. The scaler and
encoder are fitted only on the 85% training partition. The attack threshold is
selected from validation predictions by maximising F2 under a 1% validation
false-alarm ceiling. Training and threshold selection do not read KDDTest+;
the test file is used for evaluation. All knobs live
in [`config.yaml`](config.yaml).

> **Rare-class uncertainty** — validation has only 8 U2R and 149 R2L examples.
> Small changes in their predictions can move per-class F1 substantially.

---

## 📊 Results

Trained on CPU with the default [`config.yaml`](config.yaml). Evaluated on two
regimes (regenerate anytime with `python -m src.evaluate` → `reports/metrics.json`):

### In-distribution — 15% held-out validation of KDDTrain+ *(the 98%+ target regime)*

| Binary (Normal vs Attack) | Value |
|---------------------------|-------|
| Accuracy                  | **99.6%** |
| Detection rate (recall)   | **99.8%** |
| False-alarm rate          | **0.6%**  |
| ROC-AUC                   | 0.9999 |

Multi-class accuracy: **99.6%** (per-class F1 ≈ 1.00 for Normal/DOS/PROBE).

### KDDTest+ — official test set, contains novel attacks absent from training

| Binary (Normal vs Attack) | Value |
|---------------------------|-------|
| Accuracy                  | 81.7% |
| Detection rate (recall)   | 70.2% |
| Precision                 | 96.8% |
| False-alarm rate          | 3.1%  |
| ROC-AUC                   | 0.9399 |

Multi-class accuracy: 80.0%.

### Per-class F1 — where the generalisation gap actually lives

| Class | Validation F1 | KDDTest+ F1 | Δ |
|-------|--------------:|------------:|--:|
| 🟢 Normal | 0.996 | 0.820 | −0.176 |
| 🔴 DOS    | 0.999 | 0.915 | −0.084 |
| 🟡 PROBE  | 0.991 | 0.750 | −0.241 |
| 🟠 R2L    | 0.892 | **0.249** | **−0.643** |
| 🟣 U2R    | 0.737 | 0.529 | −0.208 |

**R2L has the largest generalisation gap.** Recall falls from **96.6% to 14.4%**
while precision on the few predicted R2L cases is 95.4%. The model labels 2,431
of the 2,885 R2L test records Normal. This is a material miss rate even though
the aggregate accuracy is 80.0%.

> **Why the gap?** KDDTest+ contains attack variants absent from training.
> The 99.6% validation accuracy measures performance on held-out rows from the
> same dataset, while the 80.0% test accuracy measures this harder shift. The
> current validation-selected threshold is about 0.249. Changing the threshold
> trades recall against false alarms; it does not make this model reliable for
> unseen R2L attacks or modern live traffic.

![Confusion matrix](reports/confusion_matrix.png)

### Investigating the R2L misses

Run `python -m src.r2l_diagnostics` after preprocessing and training to
regenerate the [R2L error report](reports/r2l_diagnostics.md) and its
[machine-readable results](reports/r2l_diagnostics.json). The report separates
binary-gate misses from attack-family mistakes and shows results by raw attack
subtype. It is **evaluation only**: do not select a model or threshold using
KDDTest+ outcomes.

The [R2L-weighted gate experiment](experiments/r2l_weighted_gate.py) compares
a candidate binary model with the deployed gate using the existing training
split. Its [results](experiments/r2l_weighted_gate_results.json) show a modest
KDDTest+ R2L recall improvement (14.4% to 17.5%) with the same multiclass
model. The candidate is not deployed; most R2L attacks are still missed, and
the [held-out subtype check](experiments/subtype_holdout_results.json) shows
why it should not be promoted: with `warezclient` excluded from fitting and
threshold selection, the candidate detected 0 of 890 examples (the freshly
trained current gate detected 25). The
[reproducible check](experiments/subtype_holdout.py) uses KDDTrain+ only.

## 🏭 Production notes

- **Linux / macOS** - serve with gunicorn:
  `gunicorn -w 4 -b 0.0.0.0:5000 app.app:app`.
- **Windows** - gunicorn imports `fcntl` and cannot run there.
  Use waitress instead: `python -m waitress --port=5000 app.app:app`.
- The Flask development server binds to `127.0.0.1` by default. Set `HOST`
  explicitly only when you need another interface.
- Models load **once per worker** (cached singleton) for fast inference.
- `GET /health` is suitable for uptime and load-balancer probes.
- `GET /health` returns 503 if model files are missing or cannot be loaded.
- Set `inference.attack_threshold` in `config.yaml` to override the saved
  validation-selected threshold when exploring recall/false-alarm tradeoffs.
- Before exposing the app, set `NIDS_API_KEY` and an edge request limit — see
  [Securing the endpoints](#-securing-the-endpoints). The app is an NSL-KDD
  research demo and must not be used as the sole means of intrusion detection.

---

## 🧪 Tests

```bash
pytest -q
```

The endpoint tests pass whether or not models are trained (they assert the
graceful 503 path when artifacts are missing).

---

## 🩺 Troubleshooting

<details>
<summary><strong>Predictions return 503 <code>models_not_trained</code></strong></summary>

The artifacts in `models/` are missing. `GET /health` lists exactly which ones
via `missing_artifacts`. Run `python -m src.train` to generate them.

</details>

<details>
<summary><strong><code>ModuleNotFoundError: No module named 'fcntl'</code> when starting gunicorn</strong></summary>

Gunicorn is Unix-only — `fcntl` does not exist on Windows. Use waitress
instead: `python -m waitress --port=5000 app.app:app`. See
[Production notes](#-production-notes).

</details>

<details>
<summary><strong><code>DeprecationWarning: Setting the shape on a NumPy array...</code></strong></summary>

Harmless, and **not** caused by stale artifacts — retraining will not clear it.
It originates inside joblib's own read path (`numpy_pickle.py`), which assigns
to `array.shape`, deprecated in NumPy 2.5. It fires on any joblib load
regardless of when the file was written. No fixed joblib release exists yet,
which is exactly why `numpy` is pinned to `2.5.2` in `requirements.txt` — an
unpinned upgrade could turn this warning into a hard load failure.

</details>

<details>
<summary><strong><code>FileNotFoundError</code> for KDDTrain+.txt</strong></summary>

Run `python data/download_data.py`. If the mirrors are unreachable, download
the NSL-KDD archive manually and drop `KDDTrain+.txt` and `KDDTest+.txt` into
`data/`.

</details>

<details>
<summary><strong>Changed <code>config.yaml</code> but nothing happened</strong></summary>

`src/train.py` reuses the cached `data/processed/dataset.npz` and regenerates
it if the file is missing or predates the validation split fix. If you change
encoding, scaling, or split settings, run `python -m src.preprocess` explicitly
before training again.

</details>

---

## ⚙️ Configuration

Everything tunable lives in [`config.yaml`](config.yaml): dataset paths, model
widths/dropouts, optimizer settings, early-stopping patience, class-weighting
toggle and the validation threshold rule. Retrain after changing training or
preprocessing settings; a numeric inference threshold override needs only an
app restart.

---

## ⚠️ Disclaimer

Trained on **NSL-KDD**, an academic benchmark. Excellent for learning,
demonstrations and research, but not a drop-in production sensor for modern
live networks — real deployments need current traffic, online feature
extraction and continuous retraining. Use for education and research.
