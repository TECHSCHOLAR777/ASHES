import json
from pathlib import Path

import numpy as np

from src.features.e_packer import E_VECTOR_FIELD_ORDER
from src.features.w_encoder import load_field_catalog, model_feature_layout
from src.model.arrival_eval import (
    GEOM_E,
    build_matrix,
    e_keep_indices,
    e_keep_names,
    feature_groups,
    logo_oos_proba,
    pr_auc,
    role_spans,
)


def test_e_keep_drops_only_the_two_geometric_leaks():
    names = e_keep_names()
    assert "dist_perim_m" not in names
    assert "wind_ros_ellipse_dist_m" not in names
    assert set(GEOM_E) == {"dist_perim_m", "wind_ros_ellipse_dist_m"}
    assert len(names) == len(E_VECTOR_FIELD_ORDER) - 2
    assert names == [E_VECTOR_FIELD_ORDER[i] for i in e_keep_indices()]


def test_full_w_layout_is_roles_a_to_i_and_keeps_perimeter_distance():
    layout = model_feature_layout(load_field_catalog())
    assert {row["role"] for row in layout} == set("ABCDEFGHI")
    assert layout[-1]["end"] == 200
    fields = [row["field"] for row in layout]
    assert "nearest_fire_perimeter_distance_m" in fields
    assert "surface_management_agency" in fields
    spans = role_spans(layout)
    assert set(spans) == set("ABCDEFGHI")
    # Every encoded column is in some role; no leftover strip.
    covered = np.zeros(200, dtype=bool)
    for start, end in spans.values():
        covered[start:end] = True
    assert covered.all()


def _fake_row(event_id: str, y: int, peri: float, dist_perim: float) -> dict:
    layout = model_feature_layout(load_field_catalog())
    w = np.zeros(200, dtype=float)
    mask = np.ones(200, dtype=float)
    peri_row = next(r for r in layout if r["field"] == "nearest_fire_perimeter_distance_m")
    w[peri_row["start"]] = peri
    e = np.zeros(len(E_VECTOR_FIELD_ORDER), dtype=float)
    e[E_VECTOR_FIELD_ORDER.index("dist_perim_m")] = dist_perim
    e[E_VECTOR_FIELD_ORDER.index("wind_ros_ellipse_dist_m")] = dist_perim * 0.5
    e[E_VECTOR_FIELD_ORDER.index("acres")] = 100.0
    return {
        "event_id": event_id,
        "w_vector": w.tolist(),
        "w_mask": mask.tolist(),
        "e_vector": e.tolist(),
        "spread_vector_hrrr": [12.0, 4.0, 0.2, 0.4, 0.9 if y else 0.05],
        "arrival": {"evaluable_72": True, "y_72": y},
        "_y": y,
    }


def test_matrix_keeps_full_w_and_withholds_geom_e():
    rows = [_fake_row("E1", 1, 10.0, 50.0), _fake_row("E2", 0, 8000.0, 9000.0)]
    X, y, groups, names, meta = build_matrix(rows)
    assert X.shape == (2, 200 + 13 + 5)
    assert meta["w_dim"] == 200
    assert meta["roles"] == list("ABCDEFGHI")
    assert "dist_perim_m" not in names
    assert "wind_ros_ellipse_dist_m" not in names
    assert "nearest_fire_perimeter_distance_m" in names
    assert "acres" in names
    assert names[-5:] == ["eta_hours", "eta_sigma_hours", "p_burn_24", "p_burn_48", "p_burn_72"]
    # Leak columns are recorded for diagnostics but are not in X.
    assert meta["leak_dist_perim_m"].tolist() == [50.0, 9000.0]
    peri_i = names.index("nearest_fire_perimeter_distance_m")
    assert X[0, peri_i] == 10.0
    groups_idx = feature_groups(meta["layout"], meta["e_names"], meta["w_dim"], meta["e_dim"])
    assert "W:D:nearest_fire_perimeter_distance_m" in groups_idx
    assert "E:dist_perim_m" not in groups_idx
    assert "ENG:p_burn_72" in groups_idx


def test_logo_oos_recovers_a_planted_engine_signal():
    rng = np.random.default_rng(0)
    n_events = 8
    per = 6
    y = []
    groups = []
    p72 = []
    for i in range(n_events):
        label = int(i < 4)
        for _ in range(per):
            y.append(label)
            groups.append(f"ev{i}")
            p72.append(0.8 + 0.1 * rng.random() if label else 0.05 * rng.random())
    y = np.array(y)
    groups = np.array(groups)
    X = np.column_stack([rng.normal(size=len(y)), np.array(p72)])
    oos = logo_oos_proba(X, y, groups, columns=[1])
    assert pr_auc(y, oos) > 0.9


def test_load_evaluable_requires_field_and_y72(tmp_path: Path):
    from src.model.arrival_eval import load_evaluable

    path = tmp_path / "rows.jsonl"
    good = _fake_row("G", 1, 1.0, 2.0)
    bad_no_field = dict(good, event_id="B1")
    del bad_no_field["spread_vector_hrrr"]
    bad_seed = dict(good, event_id="B2", arrival={"evaluable_72": False, "y_72": None})
    with path.open("w") as handle:
        for rec in (good, bad_no_field, bad_seed):
            rec = dict(rec)
            rec.pop("_y", None)
            handle.write(json.dumps(rec) + "\n")
    rows = load_evaluable(path)
    assert len(rows) == 1
    assert rows[0]["_y"] == 1
    assert rows[0]["event_id"] == "G"
