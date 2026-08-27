import random
from datetime import date

from src.clients.mtbs import MTBSFire
from src.model.training_data import (
    _strip_dynamic_vintage_fields,
    dist_to_ignition_centroid_m,
    ignition_datetime,
    sample_easy_negative_points,
    sample_hard_negative_points,
    sample_positive_points,
)
from src.features.w_encoder import load_field_catalog


def _square_fire(event_id="TEST1", acres=5000.0) -> MTBSFire:
    # A ~0.1-degree square centered near (39.5, -122.9), roughly matching a real MTBS fire.
    ring = [(-123.0, 39.4), (-122.8, 39.4), (-122.8, 39.6), (-123.0, 39.6), (-123.0, 39.4)]
    return MTBSFire(
        event_id=event_id,
        incident_name="TEST FIRE",
        ignition_date=date(2020, 9, 8),
        acres=acres,
        centroid_lat=39.5,
        centroid_lng=-122.9,
        geometry_rings=[[ring]],
    )


def test_ignition_datetime_pins_to_noon_utc():
    fire = _square_fire()
    dt = ignition_datetime(fire)
    assert dt.hour == 12
    assert dt.date().isoformat() == "2020-09-08"


def test_ignition_datetime_none_when_no_date():
    fire = _square_fire()
    fire.ignition_date = None
    assert ignition_datetime(fire) is None


def test_sample_positive_points_are_inside_perimeter():
    fire = _square_fire()
    rng = random.Random(42)
    points = sample_positive_points(fire, n=10, rng=rng)
    assert len(points) == 10
    for lat, lng in points:
        assert 39.4 <= lat <= 39.6
        assert -123.0 <= lng <= -122.8


def test_sample_hard_negative_points_are_outside_perimeter():
    fire = _square_fire()
    rng = random.Random(42)
    points = sample_hard_negative_points(fire, n=10, rng=rng)
    assert len(points) == 10
    from src.clients.mtbs import point_in_multipolygon

    for lat, lng in points:
        assert point_in_multipolygon(lat, lng, fire.geometry_rings) is False


def test_sample_easy_negative_points_far_from_source():
    rng = random.Random(42)
    points = sample_easy_negative_points([(39.5, -122.9)], n=5, rng=rng, min_km=100.0)
    assert len(points) == 5
    dist = dist_to_ignition_centroid_m(points[0][0], points[0][1], _square_fire())
    assert dist > 90_000  # roughly >= 100km, allowing for bearing/projection slack


def test_dist_to_ignition_centroid_zero_at_centroid():
    fire = _square_fire()
    dist = dist_to_ignition_centroid_m(fire.centroid_lat, fire.centroid_lng, fire)
    assert dist < 1.0


def test_strip_dynamic_vintage_fields_removes_ndvi_and_drought():
    catalog = load_field_catalog()
    raw_w = {
        "ndvi_current": 0.5,
        "ndvi_current_confidence": "high",
        "drought_category": "D2",
        "elevation": 500.0,  # static field, must survive
    }
    stripped = _strip_dynamic_vintage_fields(raw_w, catalog)
    assert "ndvi_current" not in stripped
    assert "ndvi_current_confidence" not in stripped
    assert "drought_category" not in stripped
    assert stripped["elevation"] == 500.0


def test_attach_spread_vector_copies_real_sample():
    from types import SimpleNamespace

    from src.model.training_data import SampleBuildResult, attach_spread_vector

    result = SampleBuildResult(
        site_id="s", event_id="e", t0="t", w_vector=[], w_mask=[], e_vector=[], y=1
    )
    spread = SimpleNamespace(
        eta_hours=9.0, eta_sigma_hours=2.0, p_burn_24=0.2, p_burn_48=0.4, p_burn_72=0.6
    )
    out = attach_spread_vector(result, spread)
    assert out.spread_vector == [9.0, 2.0, 0.2, 0.4, 0.6]
    blank = SampleBuildResult(
        site_id="s2", event_id="e", t0="t", w_vector=[], w_mask=[], e_vector=[], y=0
    )
    assert attach_spread_vector(blank, None).spread_vector is None

