import pytest

from src.model.job_c_eval import evaluate_job_c
from src.model.job_c_w_select import correlation_map
from src.model.w_allowed import allowed_column_indices
from tests.test_job_c_eval import _rows


def test_include_fields_cannot_reintroduce_leak():
    with pytest.raises(ValueError, match="not in W_allowed"):
        allowed_column_indices(include_fields=["nearest_fire_perimeter_distance_m"])


def test_include_fields_subsets_elevation_only():
    indices, names, kept = allowed_column_indices(include_fields=["elevation"])
    assert {row["field"] for row in kept} == {"elevation"}
    assert all("elevation" in n or n.endswith("_conf") for n in names)
    assert "nearest_fire_perimeter_distance_m" not in names
    assert len(indices) >= 1


def test_residual_correlation_picks_elevation_on_synthetic_same_eta():
    rows = _rows(n_events=12, n_per=12)
    cmap = correlation_map(rows, n_perm=200)
    by_field = {f["field"]: f for f in cmap["fields"]}
    assert abs(by_field["elevation"]["rho_residual"]) > 0.5
    assert "elevation" in cmap["head_fields"]
    assert "nearest_fire_perimeter_distance_m" not in cmap["head_fields"]


def test_subset_logo_still_not_a_gbm_and_kills_when_signal_is_in_w():
    rows = _rows()
    report = evaluate_job_c(rows, include_fields=["elevation"])
    assert report["protocol"]["not_a_218d_gbm"] is True
    assert report["protocol"]["w_subset"] is True
    assert report["w_allowed_fields"] == ["elevation"]
    assert report["deltas"]["w_kill_test_passed"] is True
    eng = report["scores"]["logistic_engine"]["brier"]
    both = report["scores"]["logistic_engine_w_allowed"]["brier"]
    assert both < eng - 0.01
