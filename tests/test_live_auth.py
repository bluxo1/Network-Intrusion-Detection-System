"""The benign baseline must not trigger the provisional live pilot rule."""

from datetime import datetime, timedelta, timezone

import pytest

from live_ids.auth import AuthFailure, failed_logon_alerts, parse_auth_failures


def _failure(minute: int, record_id: int) -> AuthFailure:
    return AuthFailure(
        event_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute),
        record_id=record_id,
        logon_type=2,
        status_signed=-1073741715,
        substatus_signed=-1073741718,
    )


def test_known_benign_cluster_informs_threshold():
    known_benign = [_failure(0, 1), _failure(32, 2), _failure(52, 3)]
    assert failed_logon_alerts(known_benign, threshold=4) == []
    alerts_at_three = failed_logon_alerts(known_benign, threshold=3)
    assert len(alerts_at_three) == 1
    assert alerts_at_three[0].newest_record_id == 3
    assert len(failed_logon_alerts(known_benign + [_failure(53, 4)])) == 1


def test_window_boundary_and_cooldown():
    events = [_failure(0, 1), _failure(1, 2), _failure(2, 3), _failure(3, 4)]
    events += [_failure(63, 5), _failure(64, 6), _failure(65, 7), _failure(66, 8)]
    alerts = failed_logon_alerts(events)
    assert [alert.newest_record_id for alert in alerts] == [4, 8]
    assert failed_logon_alerts(list(reversed(events))) == alerts


def test_csv_rejects_missing_timezone(tmp_path):
    csv_path = tmp_path / "auth.csv"
    csv_path.write_text(
        "time_local,event_id,record_id,logon_type,status_signed,substatus_signed\n"
        "2026-01-01T12:00:00,4625,1,2,-1073741715,-1073741718\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timezone offset"):
        parse_auth_failures(csv_path)


def test_csv_rejects_duplicate_record_that_would_inflate_alert_count(tmp_path):
    csv_path = tmp_path / "auth.csv"
    csv_path.write_text(
        "time_local,event_id,record_id,logon_type,status_signed,substatus_signed\n"
        "2026-01-01T12:00:00+00:00,4625,1,2,-1073741715,-1073741718\n"
        "2026-01-01T12:01:00+00:00,4625,1,2,-1073741715,-1073741718\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate record_id 1"):
        parse_auth_failures(csv_path)
