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

import app.app as appmod  # noqa: E402 - module, so tests can toggle its globals
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


def test_api_key_required_when_configured(client, monkeypatch):
    """With NIDS_API_KEY set, /api/predict rejects requests lacking a valid key."""
    monkeypatch.setattr(appmod, "API_KEY", "secret123")
    body = {"features": _normal_sample()}

    assert client.post("/api/predict", json=body).status_code == 401
    assert client.post("/api/predict", json=body,
                       headers={"X-API-Key": "wrong"}).status_code == 401
    # Correct key passes the auth gate; 200 if models are trained, else 503.
    ok = client.post("/api/predict", json=body, headers={"X-API-Key": "secret123"})
    assert ok.status_code in (200, 503)


def test_api_key_open_when_unset(client, monkeypatch):
    """Default (no NIDS_API_KEY) leaves the endpoint open - never 401."""
    monkeypatch.setattr(appmod, "API_KEY", "")
    resp = client.post("/api/predict", json={"features": _normal_sample()})
    assert resp.status_code != 401


def test_rate_limit_trips(monkeypatch):
    """Once the per-IP window is exceeded, further requests get 429."""
    monkeypatch.setattr(appmod, "RATE_LIMIT", 2)
    appmod._hits.clear()
    # The limiter is a no-op under TESTING, so exercise it with TESTING off.
    monkeypatch.setitem(flask_app.config, "TESTING", False)
    try:
        with flask_app.test_client() as c:
            body = {"features": _normal_sample()}
            codes = [c.post("/api/predict", json=body).status_code for _ in range(4)]
    finally:
        appmod._hits.clear()

    # First 2 pass the rate gate (200/503 depending on models); rest are 429.
    assert codes[0] != 429 and codes[1] != 429
    assert codes[2] == 429 and codes[3] == 429
