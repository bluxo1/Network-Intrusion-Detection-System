"""Synthetic Zeek connection records; no real pilot traffic is stored here."""

import json

import pytest

from live_ids.zeek import parse_conn_log


def _record(**overrides):
    row = {
        "ts": 1_700_000_000.25,
        "uid": "synthetic-1",
        "id.orig_h": "192.0.2.10",
        "id.orig_p": 50000,
        "id.resp_h": "2001:db8::20",
        "id.resp_p": 443,
        "proto": "tcp",
    }
    row.update(overrides)
    return row


def _write(tmp_path, *rows):
    path = tmp_path / "conn.log"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_parse_connections_with_optional_fields_and_utc_time(tmp_path):
    first = _record(duration=1.5, orig_bytes=100, resp_bytes=200,
                    orig_pkts=2, resp_pkts=3, service="ssl", conn_state="SF")
    second = _record(uid="synthetic-2", ts=1_700_000_001)
    path = _write(tmp_path, json.dumps(first), "", json.dumps(second))

    records = list(parse_conn_log(path, "capture-1"))

    assert len(records) == 2
    assert records[0] == {
        "event_time_utc": "2023-11-14T22:13:20.250000Z",
        "capture_id": "capture-1",
        "uid": "synthetic-1",
        "orig_h": "192.0.2.10", "orig_p": 50000,
        "resp_h": "2001:db8::20", "resp_p": 443,
        "proto": "tcp", "duration": 1.5,
        "orig_bytes": 100, "resp_bytes": 200,
        "orig_pkts": 2, "resp_pkts": 3,
        "service": "ssl", "conn_state": "SF",
    }
    assert records[1]["event_time_utc"] == "2023-11-14T22:13:21Z"
    assert records[1]["duration"] is None
    assert records[1]["service"] is None


@pytest.mark.parametrize("field,value", [
    ("ts", -1),
    ("ts", float("nan")),
    ("duration", -0.1),
    ("orig_bytes", -1),
    ("resp_pkts", 1.5),
    ("id.orig_p", True),
    ("id.resp_p", 65536),
    ("id.orig_h", "not-an-ip"),
    ("uid", ""),
])
def test_rejects_bad_fields_with_physical_line_number(tmp_path, field, value):
    path = _write(tmp_path, "", json.dumps(_record()), json.dumps(_record(**{field: value})))
    with pytest.raises(ValueError, match=f"line 3: {field}"):
        list(parse_conn_log(path, "capture-1"))


@pytest.mark.parametrize("field", [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
])
def test_rejects_missing_required_fields(tmp_path, field):
    row = _record()
    del row[field]
    path = _write(tmp_path, json.dumps(row))
    with pytest.raises(ValueError, match=f"line 1: {field} is required"):
        list(parse_conn_log(path, "capture-1"))


@pytest.mark.parametrize("text", ["not-json", "[]"])
def test_rejects_malformed_jsonl_records(tmp_path, text):
    path = _write(tmp_path, json.dumps(_record()), text)
    with pytest.raises(ValueError, match="line 2:"):
        list(parse_conn_log(path, "capture-1"))


def test_requires_capture_id(tmp_path):
    path = _write(tmp_path, json.dumps(_record()))
    with pytest.raises(ValueError, match="capture_id"):
        list(parse_conn_log(path, " "))
