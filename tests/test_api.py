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
    assert resp.status_code == (200 if models_available() else 503)
    body = resp.get_json()
    assert body["status"] == ("ok" if models_available() else "degraded")
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
    assert resp.status_code == 400
    assert client.post("/api/predict", json=["bad"]).status_code == 400


def test_api_key_required_when_configured(client, monkeypatch):
    """With NIDS_API_KEY set, /api/predict rejects requests lacking a valid key."""
    monkeypatch.setattr(appmod, "API_KEY", "secret123")
    body = {"features": _normal_sample()}

    assert client.post("/api/predict", json=body).status_code == 401
    assert client.post("/api/predict", json=body,
                       headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/predict", json=body,
                       headers={"X-API-Key": "é"}).status_code == 401
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


def test_form_requires_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(appmod, "API_KEY", "secret123")
    assert client.get("/").status_code == 401
    assert client.post("/predict", data=_normal_sample()).status_code == 401
    assert client.get("/", auth=("nids", "secret123")).status_code == 200
    expected = 200 if models_available() else 503
    assert client.post("/predict", data=_normal_sample(), auth=("nids", "secret123")).status_code == expected


def test_health_reports_model_load_failure(client, monkeypatch):
    monkeypatch.setattr(appmod, "get_predictor", lambda: (_ for _ in ()).throw(ValueError("bad model")))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.get_json()["models_ready"] is False


def test_invalid_feature_values_return_400(client):
    if not models_available():
        pytest.skip("trained artifacts required for input validation")
    for features in (
        {"src_bytes": "abc"},
        {"src_bytes": "nan"},
        {"src_bytes": "inf"},
        {"src_bytes": "1e39"},
        {"serror_rate": 1.2},
        {"src_bytes": -1},
        {"land": 2},
        {"su_attempted": 3},
        {"not_a_feature": 1},
    ):
        response = client.post("/api/predict", json={"features": features})
        assert response.status_code == 400, features
        assert response.get_json()["error"] == "invalid_features"
    assert client.post("/predict", data={"src_bytes": "nan"}).status_code == 400
    assert client.post("/api/predict", json={"features": {"su_attempted": 2}}).status_code == 200


def test_forwarded_header_cannot_bypass_rate_limit(monkeypatch):
    monkeypatch.setattr(appmod, "RATE_LIMIT", 1)
    monkeypatch.setattr(appmod, "API_KEY", "")
    monkeypatch.setitem(flask_app.config, "TESTING", False)
    appmod._hits.clear()
    try:
        with flask_app.test_client() as client:
            first = client.post("/api/predict", json={}, headers={"X-Forwarded-For": "192.0.2.1"})
            second = client.post("/api/predict", json={}, headers={"X-Forwarded-For": "192.0.2.2"})
    finally:
        appmod._hits.clear()
    assert first.status_code in (200, 503)
    assert second.status_code == 429
