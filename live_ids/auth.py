"""Minimal Windows failed-logon records and a replayable observation rule."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


AUTH_COLUMNS = (
    "time_local",
    "event_id",
    "record_id",
    "logon_type",
    "status_signed",
    "substatus_signed",
)


@dataclass(frozen=True)
class AuthFailure:
    event_time_utc: datetime
    record_id: int
    logon_type: int
    status_signed: int
    substatus_signed: int


@dataclass(frozen=True)
class AuthAlert:
    event_time_utc: datetime
    newest_record_id: int
    failures_in_window: int
    window_minutes: int


def parse_auth_failures(path: str | Path) -> list[AuthFailure]:
    """Read the minimized Windows Security 4625 export without account fields."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(AUTH_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{source.name}: missing columns: {', '.join(sorted(missing))}")
        failures = []
        seen_ids: set[int] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                try:
                    event_time = datetime.fromisoformat(row["time_local"])
                except (TypeError, ValueError):
                    raise ValueError("time_local must be an ISO 8601 timestamp") from None
                if event_time.tzinfo is None or event_time.utcoffset() is None:
                    raise ValueError("timestamp has no timezone offset")

                def integer(field: str) -> int:
                    try:
                        return int(row[field])
                    except (TypeError, ValueError):
                        raise ValueError(f"{field} must be an integer") from None

                if integer("event_id") != 4625:
                    raise ValueError("event_id must be 4625")
                record_id = integer("record_id")
                logon_type = integer("logon_type")
                status = integer("status_signed")
                substatus = integer("substatus_signed")
                if record_id <= 0 or logon_type <= 0:
                    raise ValueError("record_id and logon_type must be positive")
                if record_id in seen_ids:
                    raise ValueError(f"duplicate record_id {record_id}")
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{source.name}:{line_number}: {exc}") from exc
            seen_ids.add(record_id)
            failures.append(
                AuthFailure(
                    event_time_utc=event_time.astimezone(timezone.utc),
                    record_id=record_id,
                    logon_type=logon_type,
                    status_signed=status,
                    substatus_signed=substatus,
                )
            )
    return sorted(failures, key=lambda item: (item.event_time_utc, item.record_id))


def failed_logon_alerts(
    failures: list[AuthFailure],
    *,
    threshold: int = 4,
    window_minutes: int = 60,
    cooldown_minutes: int = 60,
) -> list[AuthAlert]:
    """Emit at most one alert per cooldown for a rolling failure count.

    This is a host-event rule for observation. It is not an ML prediction or
    evidence that an attempt was unauthorized.
    """
    if min(threshold, window_minutes, cooldown_minutes) <= 0:
        raise ValueError("threshold, window_minutes, and cooldown_minutes must be positive")
    ordered = sorted(failures, key=lambda item: (item.event_time_utc, item.record_id))
    window: deque[AuthFailure] = deque()
    alerts = []
    last_alert_at: datetime | None = None
    for failure in ordered:
        while window and failure.event_time_utc - window[0].event_time_utc >= timedelta(minutes=window_minutes):
            window.popleft()
        window.append(failure)
        if len(window) < threshold:
            continue
        if last_alert_at is not None and failure.event_time_utc - last_alert_at < timedelta(minutes=cooldown_minutes):
            continue
        alerts.append(
            AuthAlert(
                event_time_utc=failure.event_time_utc,
                newest_record_id=failure.record_id,
                failures_in_window=len(window),
                window_minutes=window_minutes,
            )
        )
        last_alert_at = failure.event_time_utc
    return alerts
