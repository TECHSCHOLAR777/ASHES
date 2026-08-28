import numpy as np

from src.features.w_encoder import load_field_catalog, model_feature_layout
from src.model.job_c_eval import evaluate_job_c
from src.model.w_allowed import allowed_column_indices


def _rows(n_events: int = 10, n_per: int = 16):
    layout = model_feature_layout(load_field_catalog())
    elev = next(r for r in layout if r["field"] == "elevation")
    peri = next(r for r in layout if r["field"] == "nearest_fire_perimeter_distance_m")
    indices, _, _ = allowed_column_indices()
    rows = []
    rng = np.random.default_rng(0)
    for e in range(n_events):
        for i in range(n_per):
            y = int((e + i) % 2 == 0)
            # Same-ETA clock: everyone is ~18 h out. Engine p72 is 1 for all.
            eta = 18.0 + float(rng.normal(0, 0.4))
            w = np.zeros(200, dtype=float)
            mask = np.ones(200, dtype=float)
            w[elev["start"]] = 800.0 if y else 200.0
            w[peri["start"]] = 10.0 if y else 50000.0  # leak column; head must ignore it
            rows.append(
                {
                    "event_id": f"E{e}",
                    "event": e,
                    "w_vector": w.tolist(),
                    "w_mask": mask.tolist(),
                    "_y": y,
                    "_spread": [eta, 0.0, 1.0, 1.0, 1.0],
                    "engine": "elmfire_2025.0212",
                    "arrival": {"evaluable_72": True, "y_72": y},
                }
            )
    assert elev["start"] in indices
    assert peri["start"] not in indices
    return rows


def test_engine_plus_w_allowed_beats_engine_only_when_residual_is_in_w():
    report = evaluate_job_c(_rows())
    assert report["protocol"]["not_a_218d_gbm"] is True
    assert report["n_events"] == 10
    assert report["n_pos"] > 20
    eng = report["scores"]["logistic_engine"]["brier"]
    both = report["scores"]["logistic_engine_w_allowed"]["brier"]
    shuffled = report["scores"]["logistic_engine_shuffled_w"]["brier"]
    assert both < eng - 0.01
    assert shuffled > both
    assert report["deltas"]["w_kill_test_passed"] is True
    assert "nearest_fire_perimeter_distance_m" not in report["w_allowed_fields"]
    slices = report["same_eta_slices"]
    mid = [s for s in slices if s["eta_lo"] <= 18 < s["eta_hi"]]
    assert mid
    assert mid[0]["n"] >= 20
