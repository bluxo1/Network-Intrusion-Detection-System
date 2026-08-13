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
"""

import json
import os
import sys

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
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        return render_template("result.html", error=f"Prediction failed: {exc}",
                               class_style=CLASS_STYLE), 500

    style = CLASS_STYLE.get(result["predicted_class"], CLASS_STYLE["Normal"])
    return render_template(
        "result.html", result=result, style=style, features=features,
        class_style=CLASS_STYLE,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint. Body: {"features": {<feature>: <value>, ...}} or a bare dict."""
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
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": "prediction_failed", "detail": str(exc)}), 500
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
