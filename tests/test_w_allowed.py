from src.features.w_encoder import load_field_catalog, model_feature_layout
from src.model.w_allowed import allowed_column_indices, allowed_fields, excluded_fields


def test_job_c_head_drops_leaks_and_t0_fuel_not_role_j():
    catalog = load_field_catalog()
    skip = excluded_fields(catalog)
    assert "nearest_fire_perimeter_distance_m" in skip
    assert "ndvi_current" in skip
    assert "lcms_class" in skip
    assert "tree_canopy_pct" in skip
    assert "most_recent_burn_year" in skip
    assert "drought_category" in skip
    allowed = set(allowed_fields(catalog))
    assert "elevation" in allowed
    assert "land_use_class" in allowed
    assert "nearest_road_distance_m" in allowed
    assert "lightning_annual_flash_days" in allowed
    assert "nearest_fire_perimeter_distance_m" not in allowed
    indices, names, kept = allowed_column_indices(catalog)
    layout = model_feature_layout(catalog)
    full = layout[-1]["end"]
    assert full == 200
    assert len(indices) < 200
    assert len(indices) == len(names)
    assert all(0 <= i < 200 for i in indices)
    assert "nearest_fire_perimeter_distance_m" not in names
    assert not any(n.startswith("ndvi_current") for n in names)
    assert {row["field"] for row in kept}.isdisjoint(skip)
