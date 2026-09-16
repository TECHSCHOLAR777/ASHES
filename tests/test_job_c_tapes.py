from src.clients.timed_perimeters import _snapshots_from_wfigs_features, _collapse_same_timestamp
from src.model.arrival_labels import last_rings, seed_rings
from src.clients.timed_perimeters import TimedFireSeries, TimedSnapshot
from datetime import datetime, timezone, timedelta


def test_wfigs_feature_parser_uses_unique_fire_identifier():
    t = 1_720_000_000_000
    feat = {
        "attributes": {
            "attr_UniqueFireIdentifier": "2024-CA-XYZ-0001",
            "poly_IRWINID": "{AAAA}",
            "poly_IncidentName": "Test Fire",
            "poly_PolygonDateTime": t,
            "poly_GISAcres": 12.5,
            "poly_MapMethod": "IR Image Interpretation",
        },
        "geometry": {"rings": [[[-120.0, 40.0], [-119.9, 40.0], [-119.9, 40.1], [-120.0, 40.1], [-120.0, 40.0]]]},
    }
    snaps = _snapshots_from_wfigs_features([feat])
    assert len(snaps) == 1
    assert snaps[0].fire_id == "2024-CA-XYZ-0001"
    assert snaps[0].map_method == "IR Image Interpretation"


def test_collapse_prefers_ir_over_sketch_at_same_time():
    t0 = datetime(2024, 7, 1, tzinfo=timezone.utc)
    ring = [[(-120.0, 40.0), (-119.9, 40.0), (-119.9, 40.1), (-120.0, 40.1), (-120.0, 40.0)]]
    snaps = [
        TimedSnapshot(t0, "F", "n", 10, "Hand Sketch", ring, "wfigs_daily"),
        TimedSnapshot(t0, "F", "n", 9, "IR Image Interpretation", ring, "wfigs_daily"),
        TimedSnapshot(t0 + timedelta(hours=24), "F", "n", 20, "Mixed Methods", ring, "wfigs_daily"),
    ]
    out = _collapse_same_timestamp(snaps)
    assert len(out) == 2
    assert "IR" in (out[0].map_method or "")


def test_last_rings_are_not_the_seed():
    t0 = datetime(2024, 7, 1, tzinfo=timezone.utc)
    small = [[(-120.0, 40.0), (-119.99, 40.0), (-119.99, 40.01), (-120.0, 40.01), (-120.0, 40.0)]]
    big = [[(-121.0, 39.0), (-119.0, 39.0), (-119.0, 41.0), (-121.0, 41.0), (-121.0, 39.0)]]
    series = TimedFireSeries(
        "F",
        "n",
        "wfigs_daily",
        [
            TimedSnapshot(t0, "F", "n", 1, "IR", small, "wfigs_daily"),
            TimedSnapshot(t0 + timedelta(hours=80), "F", "n", 90, "IR", big, "wfigs_daily"),
        ],
    )
    assert seed_rings(series) == small
    assert last_rings(series) == big
    assert seed_rings(series) != last_rings(series)
