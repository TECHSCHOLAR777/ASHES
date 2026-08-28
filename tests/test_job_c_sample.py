from datetime import datetime, timedelta, timezone
from random import Random

from src.clients.timed_perimeters import TimedFireSeries, TimedSnapshot
from src.model.job_c_sample import early_tape_ok, labeled_rows, sample_series


def _square(lng0: float, lat0: float, half: float) -> list[list[tuple[float, float]]]:
    return [[
        (lng0 - half, lat0 - half),
        (lng0 + half, lat0 - half),
        (lng0 + half, lat0 + half),
        (lng0 - half, lat0 + half),
        (lng0 - half, lat0 - half),
    ]]


def test_annulus_points_are_outside_seed_and_labelable():
    t0 = datetime(2024, 7, 1, 12, tzinfo=timezone.utc)
    snaps = [
        TimedSnapshot(t0, "2024-CA-X", "t", 10, "Infrared Image", _square(-120.0, 40.0, 0.02), "wfigs_daily"),
        TimedSnapshot(t0 + timedelta(hours=24), "2024-CA-X", "t", 40, "Infrared Image", _square(-120.0, 40.0, 0.05), "wfigs_daily"),
        TimedSnapshot(t0 + timedelta(hours=80), "2024-CA-X", "t", 90, "Infrared Image", _square(-120.0, 40.0, 0.12), "wfigs_daily"),
        TimedSnapshot(t0 + timedelta(hours=120), "2024-CA-X", "t", 120, "Infrared Image", _square(-120.0, 40.0, 0.18), "wfigs_daily"),
    ]
    series = TimedFireSeries("2024-CA-X", "t", "wfigs_daily", snaps)
    assert early_tape_ok(series) is True
    sites = sample_series(series, Random(0), n_annulus=12, n_hard=12, n_far=6)
    assert len(sites) >= 20
    roles = {s.role for s in sites}
    assert roles == {"annulus", "hard_neg", "far_neg"}
    rows = labeled_rows(series, sites)
    assert all(not r["arrival"]["already_burned_at_seed"] for r in rows)
    eval_rows = [r for r in rows if r["arrival"]["evaluable_72"]]
    assert len(eval_rows) >= 15
    annulus = [r for r in rows if r["sample_role"] == "annulus" and r["arrival"]["evaluable_72"]]
    assert annulus
    assert all(r["arrival"]["y_72"] == 1 for r in annulus)
    hard = [r for r in rows if r["sample_role"] in {"hard_neg", "far_neg"} and r["arrival"]["evaluable_72"]]
    assert hard
    assert all(r["arrival"]["y_72"] == 0 for r in hard)


def test_sparse_early_tape_is_rejected():
    t0 = datetime(2024, 7, 1, 12, tzinfo=timezone.utc)
    snaps = [
        TimedSnapshot(t0, "F", "t", 10, "Hand Sketch", _square(-120.0, 40.0, 0.02), "wfigs_daily"),
        TimedSnapshot(t0 + timedelta(hours=200), "F", "t", 80, "Hand Sketch", _square(-120.0, 40.0, 0.2), "wfigs_daily"),
    ]
    series = TimedFireSeries("F", "t", "wfigs_daily", snaps)
    assert early_tape_ok(series, 96.0) is False
