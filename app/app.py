"""
Flask web application for the NIDS.

Routes
------
GET  /             - render the 41-feature input form
POST /predict      - classify submitted form data, render the result page
POST /api/predict  - JSON in / JSON out (for programmatic consumers)
GET  /health       - liveness + model-availability probe

Run locally:   python app/app.py
Production:    gunicorn -w 4 -b 0.0.0.0:5000 app.app:app

Security (opt-in, off by default so local demos keep working)
-------------------------------------------------------------
* NIDS_API_KEY   - if set, /api/predict requires an ``X-API-Key`` header that
                   matches it (constant-time compared). Unset => open, as before.
* NIDS_RATE_LIMIT- max requests/minute/client-IP for the inference endpoints.
                   Defaults to 120; set to 0 to disable. Skipped under TESTING.
"""

import hmac
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque

# Make the project root importable whether launched as `python app/app.py`
# or `gunicorn app.app:app` from the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, render_template, request  # noqa: E402

from app.preprocessor import build_field_specs, parse_form  # noqa: E402
from app.predictor import get_predictor, missing_artifacts, models_available  # noqa: E402
from src.config import CONFIG, abspath  # noqa: E402
from src.schema import CATEGORICAL_COLUMNS, CLASS_NAMES  # noqa: E402

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

logger = logging.getLogger("nids.app")

# --- Opt-in hardening for the inference endpoints -------------------------
# Read once at import. Both features are inert unless configured, so existing
# local/demo usage and the test-suite are unaffected.
API_KEY = os.environ.get("NIDS_API_KEY", "").strip()
try:
    RATE_LIMIT = int(os.environ.get("NIDS_RATE_LIMIT", "120"))  # req/min/IP; 0 disables
except ValueError:
    RATE_LIMIT = 120
_RATE_WINDOW_SEC = 60.0
_hits: "defaultdict[str, deque]" = defaultdict(deque)


def _client_ip() -> str:
    """Best-effort client IP; trusts the first X-Forwarded-For hop if present."""
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limited(ip: str) -> bool:
    """Fixed-window-per-IP limiter. No-op under TESTING or when disabled."""
    if RATE_LIMIT <= 0 or app.config.get("TESTING"):
        return False
    now = time.monotonic()
    dq = _hits[ip]
    while dq and now - dq[0] > _RATE_WINDOW_SEC:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return True
    dq.append(now)
    return False


def _auth_ok() -> bool:
    """True if auth is disabled, or the presented X-API-Key matches (constant-time)."""
    if not API_KEY:
        return True
    presented = request.headers.get("X-API-Key", "")
    return hmac.compare_digest(presented, API_KEY)

# Per-class colour + emoji used for result styling in the templates.
CLASS_STYLE = {
    "Normal": {"color": "#1a9850", "emoji": "\U0001F7E2", "desc": "Benign traffic - no threat detected."},
    "DOS": {"color": "#d73027", "emoji": "\U0001F534", "desc": "Denial of Service - flooding to exhaust resources."},
    "PROBE": {"color": "#f6c343", "emoji": "\U0001F7E1", "desc": "Probe / scan - reconnaissance of hosts and ports."},
    "R2L": {"color": "#fc8d59", "emoji": "\U0001F7E0", "desc": "Remote-to-Local - unauthorized remote access attempt."},
    "U2R": {"color": "#8e44ad", "emoji": "\U0001F7E3", "desc": "User-to-Root - privilege escalation to superuser."},
}

# Fallback categories so the form still renders before the model is trained.
FALLBACK_CATEGORIES = {
    "protocol_type": ["tcp", "udp", "icmp"],
    "flag": ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH", "RSTOS0"],
    "service": [
        "http", "smtp", "ftp", "ftp_data", "telnet", "ssh", "domain_u", "private",
        "other", "eco_i", "ecr_i", "finger", "auth", "pop_3", "imap4", "ntp_u",
        "netbios_ns", "netbios_dgm", "netbios_ssn", "http_443", "X11", "IRC",
    ],
}


def load_metadata() -> dict:
    """Load model metadata if trained; otherwise return a usable fallback."""
    meta_path = abspath(CONFIG["artifacts"]["metadata"])
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categories": FALLBACK_CATEGORIES,
        "class_names": CLASS_NAMES,
    }


@app.route("/", methods=["GET"])
def index():
    metadata = load_metadata()
    groups = build_field_specs(metadata)
    return render_template(
        "index.html",
        groups=groups,
        models_ready=models_available(),
        class_style=CLASS_STYLE,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if _rate_limited(_client_ip()):
        return render_template(
            "result.html", error="Too many requests. Please slow down and try again.",
            class_style=CLASS_STYLE,
        ), 429

    if not models_available():
        return render_template(
            "result.html", error=(
                "Models are not trained yet. Run `python -m src.train` to "
                "generate the model artifacts, then try again."
            ), class_style=CLASS_STYLE,
        ), 503

    features = parse_form(request.form)
    try:
        result = get_predictor().predict_features(features)
    except Exception:  # noqa: BLE001 - log detail server-side, stay generic to the client
        logger.exception("prediction failed on /predict")
        return render_template("result.html",
                               error="Prediction failed due to an internal error.",
                               class_style=CLASS_STYLE), 500

    style = CLASS_STYLE.get(result["predicted_class"], CLASS_STYLE["Normal"])
    return render_template(
        "result.html", result=result, style=style, features=features,
        class_style=CLASS_STYLE,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint. Body: {"features": {<feature>: <value>, ...}} or a bare dict."""
    if _rate_limited(_client_ip()):
        return jsonify({"error": "rate_limited", "detail": "too many requests"}), 429
    if not _auth_ok():
        return jsonify({"error": "unauthorized"}), 401

    if not models_available():
        return jsonify({
            "error": "models_not_trained",
            "missing_artifacts": missing_artifacts(),
        }), 503

    payload = request.get_json(silent=True) or {}
    features = payload.get("features", payload)
    if not isinstance(features, dict):
        return jsonify({"error": "invalid_payload", "detail": "expected a JSON object of features"}), 400

    try:
        result = get_predictor().predict_features(features)
    except Exception:  # noqa: BLE001 - log detail server-side, stay generic to the client
        logger.exception("prediction failed on /api/predict")
        return jsonify({"error": "prediction_failed"}), 500
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_ready": models_available(),
        "missing_artifacts": missing_artifacts(),
        "classes": CLASS_NAMES,
    })


if __name__ == "__main__":
    # Dev server. In production use gunicorn (see the module docstring / README).
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
