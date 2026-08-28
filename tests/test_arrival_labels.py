from datetime import datetime, timedelta, timezone

from src.clients.timed_perimeters import TimedFireSeries, TimedSnapshot
from src.model.arrival_labels import label_point


def _square(lng0: float, lat0: float, half: float = 0.05) -> list[list[tuple[float, float]]]:
    return [[
        (lng0 - half, lat0 - half),
        (lng0 + half, lat0 - half),
        (lng0 + half, lat0 + half),
        (lng0 - half, lat0 + half),
        (lng0 - half, lat0 - half),
    ]]


def _series(snaps: list[TimedSnapshot]) -> TimedFireSeries:
    return TimedFireSeries(fire_id="F1", name="test", source="unit", snapshots=snaps)


def test_label_excludes_already_burned_at_seed():
    t0 = datetime(2020, 8, 1, 12, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=30)
    series = _series(
        [
            TimedSnapshot(t0, "F1", "test", 10, "Infrared Image", _square(-120.0, 40.0), "unit"),
            TimedSnapshot(t1, "F1", "test", 40, "Infrared Image", _square(-120.0, 40.0, 0.2), "unit"),
        ]
    )
    lab = label_point(40.0, -120.0, series)
    assert lab.already_burned_at_seed is True
    assert lab.evaluable_72 is False
    assert lab.y_72 is None


def test_label_positive_arrives_between_snapshots():
    t0 = datetime(2020, 8, 1, 12, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=30)
    small = _square(-120.0, 40.0, 0.02)
    big = _square(-120.0, 40.0, 0.2)
    series = _series(
        [
            TimedSnapshot(t0, "F1", "test", 10, "Infrared Image", small, "unit"),
            TimedSnapshot(t1, "F1", "test", 80, "Infrared Image", big, "unit"),
        ]
    )
    # Site is outside the seed square, inside the later square.
    lab = label_point(40.12, -120.0, series)
    assert lab.already_burned_at_seed is False
    assert lab.arrival_hours == 30.0
    assert lab.y_24 == 0
    assert lab.y_48 == 1
    assert lab.y_72 == 1
    assert lab.evaluable_72 is True


def test_label_negative_requires_horizon_coverage():
    t0 = datetime(2020, 8, 1, 12, tzinfo=timezone.utc)
    t_short = t0 + timedelta(hours=10)
    t_long = t0 + timedelta(hours=80)
    small = _square(-120.0, 40.0, 0.02)
    still_small = _square(-120.0, 40.0, 0.03)
    short_series = _series(
        [
            TimedSnapshot(t0, "F1", "test", 10, "Hand Sketch", small, "unit"),
            TimedSnapshot(t_short, "F1", "test", 12, "Hand Sketch", still_small, "unit"),
        ]
    )
    short = label_point(41.0, -121.0, short_series)
    assert short.y_24 is None  # tape shorter than 24 h and never hit
    assert short.evaluable_72 is False

    long_series = _series(
        [
            TimedSnapshot(t0, "F1", "test", 10, "Infrared Image", small, "unit"),
            TimedSnapshot(t_long, "F1", "test", 12, "Infrared Image", still_small, "unit"),
        ]
    )
    long = label_point(41.0, -121.0, long_series)
    assert long.y_24 == 0
    assert long.y_48 == 0
    assert long.y_72 == 0
    assert long.evaluable_72 is True


def test_single_snapshot_is_not_a_time_series():
    t0 = datetime(2020, 8, 1, 12, tzinfo=timezone.utc)
    series = _series(
        [TimedSnapshot(t0, "F1", "test", 10, "Infrared Image", _square(-120.0, 40.0), "unit")]
    )
    lab = label_point(40.0, -120.0, series)
    assert lab.n_times == 1
    assert lab.evaluable_72 is False
    assert lab.y_72 is None
