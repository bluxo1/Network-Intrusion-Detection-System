"""
Endpoint tests for the Flask app.

These are written to pass whether or not the models have been trained yet:
  * /health and /  always work.
  * /predict and /api/predict return a valid prediction when artifacts exist,
    and a well-formed 503 when they do not.

Run with:  pytest -q
"""

import os
import sys

import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.app import app as flask_app  # noqa: E402
from app.predictor import models_available  # noqa: E402
from src.schema import CLASS_NAMES  # noqa: E402


@pytest.fixture()
def client():
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as c:
        yield c


def _normal_sample():
    return {
        "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 215, "dst_bytes": 45076, "logged_in": 1,
        "count": 1, "srv_count": 1, "same_srv_rate": 1.0,
        "dst_host_count": 9, "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
    }


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["classes"] == CLASS_NAMES


def test_index_renders_form(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Network Intrusion Detection" in resp.data
    assert b"Analyze Traffic" in resp.data


def test_api_predict(client):
    resp = client.post("/api/predict", json={"features": _normal_sample()})
    if not models_available():
        assert resp.status_code == 503
        assert resp.get_json()["error"] == "models_not_trained"
        return
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["predicted_class"] in CLASS_NAMES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["class_probabilities"]) == set(CLASS_NAMES)


def test_api_predict_rejects_bad_payload(client):
    resp = client.post("/api/predict", json={"features": "not-a-dict"})
    # 503 first if models missing; otherwise a 400 for the bad payload.
    assert resp.status_code in (400, 503)
