"""Replay Windows auth and Zeek capture metadata without exposing raw records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from live_ids.auth import failed_logon_alerts, parse_auth_failures
from live_ids.zeek import parse_conn_log


def _load_labels(path: str | Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"record_id", "label"}.issubset(reader.fieldnames or ()):
            raise ValueError("label CSV needs record_id and label columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                record_id = int(row["record_id"])
                label = row["label"].strip().lower()
                if record_id <= 0 or label not in {"normal", "attack", "unknown"}:
                    raise ValueError("invalid record_id or label")
                if record_id in labels:
                    raise ValueError("duplicate record_id")
            except (TypeError, ValueError, AttributeError) as exc:
                reason = "duplicate record_id" if str(exc) == "duplicate record_id" else "invalid record_id or label"
                raise ValueError(f"label CSV line {line_number}: {reason}") from exc
            labels[record_id] = label
    return labels


def replay(
    auth_csv: str | Path,
    *,
    conn_log: str | Path | None = None,
    capture_id: str | None = None,
    labels_csv: str | Path | None = None,
) -> dict:
    """Return a privacy-minimized replay report for a local pilot."""
    failures = parse_auth_failures(auth_csv)
    alerts = failed_logon_alerts(failures)
    report: dict = {
        "schema_version": "live-pilot-v1",
        "mode": "observation_only",
        "auth_failures": len(failures),
        "auth_candidate_alerts": len(alerts),
        "auth_rule": {"threshold": 4, "window_minutes": 60, "cooldown_minutes": 60},
        "alerts": [
            {
                "event_time_utc": alert.event_time_utc.isoformat().replace("+00:00", "Z"),
                "newest_record_id": alert.newest_record_id,
                "failures_in_window": alert.failures_in_window,
            }
            for alert in alerts
        ],
    }
    if labels_csv is not None:
        labels = _load_labels(labels_csv)
        record_ids = {failure.record_id for failure in failures}
        extra = set(labels) - record_ids
        if extra:
            raise ValueError(f"label CSV contains {len(extra)} record IDs outside auth CSV")
        report["auth_label_counts"] = dict(
            sorted(Counter(labels.get(failure.record_id, "unknown") for failure in failures).items())
        )
        report["unlabeled_auth_failures"] = sum(failure.record_id not in labels for failure in failures)
    if conn_log is not None:
        if not capture_id:
            raise ValueError("capture_id is required with conn_log")
        seen_uids: set[str] = set()
        count = 0
        for connection in parse_conn_log(conn_log, capture_id):
            if connection["uid"] in seen_uids:
                raise ValueError(f"duplicate Zeek uid in capture {capture_id}")
            seen_uids.add(connection["uid"])
            count += 1
        report["zeek_capture_id"] = capture_id
        report["zeek_connections"] = count
        report["zeek_schema_version"] = "zeek-conn-v1"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser("replay", help="replay local pilot records")
    replay_parser.add_argument("--auth", required=True, help="minimized 4625 CSV export")
    replay_parser.add_argument("--labels", help="reviewed record_id,label CSV")
    replay_parser.add_argument("--conn", help="Zeek JSON conn.log")
    replay_parser.add_argument("--capture-id", help="identifier for --conn capture")
    args = parser.parse_args(argv)
    try:
        result = replay(
            args.auth,
            conn_log=args.conn,
            capture_id=args.capture_id,
            labels_csv=args.labels,
        )
    except OSError as exc:
        name = Path(exc.filename).name if exc.filename else "input file"
        print(f"replay failed: cannot read {name}: {exc.strerror or type(exc).__name__}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"replay failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
