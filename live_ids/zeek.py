"""Read Zeek JSON ``conn.log`` records for the separate live pilot.

``parse_conn_log`` streams one normalized dictionary per connection. It does
not create NSL-KDD features or call the demo model. The caller supplies a
capture ID so records from different offline captures can be kept separate.
"""

from __future__ import annotations

import ipaddress
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_OPTIONAL_TEXT = ("service", "conn_state")
_OPTIONAL_COUNTS = ("orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts")


def _invalid(line: int, field: str, reason: str) -> ValueError:
    return ValueError(f"conn.log line {line}: {field} {reason}")


def _number(record: dict, name: str, line: int, *, integer: bool = False,
            required: bool = False) -> int | float | None:
    if name not in record or record[name] is None:
        if required:
            raise _invalid(line, name, "is required")
        return None
    value = record[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(line, name, "must be a non-negative number")
    if integer and not isinstance(value, int):
        raise _invalid(line, name, "must be a non-negative integer")
    if value < 0 or (isinstance(value, float) and not math.isfinite(value)):
        raise _invalid(line, name, "must be finite and non-negative")
    return value


def _text(record: dict, name: str, line: int, *, required: bool = False) -> str | None:
    if name not in record or record[name] is None:
        if required:
            raise _invalid(line, name, "is required")
        return None
    value = record[name]
    if not isinstance(value, str) or not value.strip():
        raise _invalid(line, name, "must be a non-empty string")
    return value


def _address(record: dict, name: str, line: int) -> str:
    value = _text(record, name, line, required=True)
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise _invalid(line, name, "must be an IP address") from exc
    return value


def _port(record: dict, name: str, line: int) -> int:
    value = _number(record, name, line, integer=True, required=True)
    if value > 65535:
        raise _invalid(line, name, "must be between 0 and 65535")
    return value


def parse_conn_log(path: str | Path, capture_id: str) -> Iterator[dict]:
    """Yield validated connection dictionaries from Zeek JSON Lines.

    Required fields are ``ts``, ``uid``, ``id.orig_h``, ``id.orig_p``,
    ``id.resp_h``, ``id.resp_p`` and ``proto``. Optional fields are
    ``duration``, byte/packet counts, ``service`` and ``conn_state``; absent
    optional fields are returned as ``None``. A malformed record raises
    ``ValueError`` identifying its physical line number. Blank lines are
    ignored. ``event_time_utc`` is an ISO 8601 UTC timestamp ending in ``Z``.
    """
    if not isinstance(capture_id, str) or not capture_id.strip():
        raise ValueError("capture_id must be a non-empty string")

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise _invalid(line_number, "JSON", "is malformed") from exc
            if not isinstance(record, dict):
                raise _invalid(line_number, "record", "must be a JSON object")

            ts = _number(record, "ts", line_number, required=True)
            try:
                event_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (OverflowError, OSError, ValueError) as exc:
                raise _invalid(line_number, "ts", "is outside the supported range") from exc

            normalized = {
                "event_time_utc": event_time.isoformat().replace("+00:00", "Z"),
                "capture_id": capture_id,
                "uid": _text(record, "uid", line_number, required=True),
                "orig_h": _address(record, "id.orig_h", line_number),
                "orig_p": _port(record, "id.orig_p", line_number),
                "resp_h": _address(record, "id.resp_h", line_number),
                "resp_p": _port(record, "id.resp_p", line_number),
                "proto": _text(record, "proto", line_number, required=True),
                "duration": _number(record, "duration", line_number),
            }
            for name in _OPTIONAL_COUNTS:
                normalized[name] = _number(record, name, line_number, integer=True)
            for name in _OPTIONAL_TEXT:
                normalized[name] = _text(record, name, line_number)
            yield normalized
