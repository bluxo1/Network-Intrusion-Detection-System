"""End-to-end replay keeps live records away from the NSL-KDD predictor."""

import json

from live_ids.cli import main, replay


def test_replay_minimized_auth_and_zeek_inputs(tmp_path):
    auth = tmp_path / "auth.csv"
    auth.write_text(
        "time_local,event_id,record_id,logon_type,status_signed,substatus_signed\n"
        "2026-01-01T10:00:00+05:30,4625,10,2,-1073741715,-1073741718\n"
        "2026-01-01T10:20:00+05:30,4625,11,2,-1073741715,-1073741718\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("record_id,label\n10,normal\n11,normal\n", encoding="utf-8")
    conn = tmp_path / "conn.log"
    conn.write_text(
        json.dumps({
            "ts": 1767232800.0, "uid": "synthetic-conn", "id.orig_h": "192.0.2.1",
            "id.orig_p": 50000, "id.resp_h": "192.0.2.2", "id.resp_p": 443,
            "proto": "tcp",
        }) + "\n",
        encoding="utf-8",
    )

    report = replay(auth, labels_csv=labels, conn_log=conn, capture_id="synthetic")

    assert report["auth_failures"] == 2
    assert report["auth_candidate_alerts"] == 0
    assert report["auth_label_counts"] == {"normal": 2}
    assert report["zeek_connections"] == 1
    assert "192.0.2.1" not in json.dumps(report)


def test_replay_error_does_not_print_parent_directory(tmp_path, capsys):
    private_dir = tmp_path / "private-account-name"
    private_dir.mkdir()
    auth = private_dir / "auth.csv"
    auth.write_text(
        "time_local,event_id,record_id,logon_type,status_signed,substatus_signed\n"
        "missing-offset,4625,1,2,-1073741715,-1073741718\n",
        encoding="utf-8",
    )

    assert main(["replay", "--auth", str(auth)]) == 2
    error = capsys.readouterr().err
    assert "auth.csv:2" in error
    assert "private-account-name" not in error
    assert "missing-offset" not in error
