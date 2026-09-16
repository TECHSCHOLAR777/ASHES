from src.features.w_encoder import load_field_catalog, model_feature_layout


def test_model_feature_layout_covers_roles_a_to_i_and_conf_bits():
    layout = model_feature_layout(load_field_catalog())
    roles = {row["role"] for row in layout}
    assert roles == set("ABCDEFGHI")
    assert all(row["end"] > row["start"] for row in layout)
    assert layout[-1]["end"] == 200  # encoded W width used by V1/V2 pickles
    fields = [row["field"] for row in layout]
    assert "nearest_fire_perimeter_distance_m" in fields
    assert "surface_management_agency" in fields
