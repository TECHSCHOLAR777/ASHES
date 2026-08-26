from datetime import datetime, timezone

from src.features.w_encoder import encode_w, ordered_model_fields


def test_missing_float_field_masks_zero():
    raw = {}  # elevation entirely absent
    result = encode_w(raw)
    fields = ordered_model_fields()
    idx = [i for i, (r, n, m) in enumerate(fields) if n == "elevation"][0]
    # Each field occupies at least 2 slots (value + _conf); locate the elevation value slot
    # by name in feature_names rather than assuming a fixed offset.
    value_pos = result.feature_names.index("elevation")
    assert result.mask[value_pos] == 0.0
    assert result.vector[value_pos] == 0.0


def test_present_float_field_masks_one():
    raw = {"elevation": 500.0}
    result = encode_w(raw)
    value_pos = result.feature_names.index("elevation")
    assert result.mask[value_pos] == 1.0
    assert result.vector[value_pos] == 500.0  # identity scaler by default


def test_categorical_one_hot_known_value():
    raw = {"lcms_class": "forest"}
    result = encode_w(raw)
    pos = result.feature_names.index("lcms_class__forest")
    assert result.vector[pos] == 1.0
    other_pos = result.feature_names.index("lcms_class__other")
    assert result.vector[other_pos] == 0.0


def test_categorical_one_hot_unknown_value_goes_to_other():
    raw = {"lcms_class": "some_unseen_class"}
    result = encode_w(raw)
    other_pos = result.feature_names.index("lcms_class__other")
    assert result.vector[other_pos] == 1.0


def test_most_recent_burn_year_encoded_as_years_since_burn():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raw = {"most_recent_burn_year": 2020}
    result = encode_w(raw, now=now)
    pos = result.feature_names.index("years_since_burn")
    assert result.vector[pos] == 6.0
    assert result.mask[pos] == 1.0


def test_most_recent_burn_year_null_imputes_conservative_50():
    raw = {}
    result = encode_w(raw)
    pos = result.feature_names.index("years_since_burn")
    assert result.vector[pos] == 50.0
    assert result.mask[pos] == 0.0


def test_confidence_bit_high_sets_one():
    raw = {"elevation": 500.0, "elevation_confidence": "high"}
    result = encode_w(raw)
    pos = result.feature_names.index("elevation_conf")
    assert result.vector[pos] == 1.0


def test_confidence_bit_low_sets_zero():
    raw = {"elevation": 500.0, "elevation_confidence": "low"}
    result = encode_w(raw)
    pos = result.feature_names.index("elevation_conf")
    assert result.vector[pos] == 0.0


def test_ordered_categorical_drought_category():
    raw = {"drought_category": "D3"}
    result = encode_w(raw)
    pos = result.feature_names.index("drought_category")
    assert result.vector[pos] == 4.0  # index of D3 in [None, D0, D1, D2, D3, D4]


def test_role_j_fields_never_in_vector():
    raw = {"nearest_airport_name": "TEST", "nearest_airport_distance_m": 1000.0}
    result = encode_w(raw)
    assert "nearest_airport_distance_m" not in result.feature_names


def test_join_key_fields_never_in_vector():
    raw = {"nearest_fire_station_name": "Station 5"}
    result = encode_w(raw)
    assert "nearest_fire_station_name" not in result.feature_names


def test_citations_collected_when_source_url_present():
    raw = {"elevation": 500.0, "elevation_source_url": "https://mireye.example/elevation", "fetched_at": "2026-08-27T00:00:00Z"}
    result = encode_w(raw)
    assert any(c["field"] == "elevation" for c in result.citations)
