from src.model.job_c_eval import evaluate_job_c
from tests.test_job_c_eval import _rows


def test_gbm_calibrator_kills_when_residual_is_in_w_and_ablation_keeps_elevation():
    report = evaluate_job_c(_rows(), estimator="gbm", ablate=True)
    assert report["estimator"] == "gbm"
    assert report["protocol"]["not_a_ranking_gbm"] is True
    assert report["protocol"]["not_a_218d_gbm"] is False
    eng = report["scores"]["gbm_engine"]["brier"]
    both = report["scores"]["gbm_engine_w_allowed"]["brier"]
    shuffled = report["scores"]["gbm_engine_shuffled_w"]["brier"]
    assert both < eng - 0.01
    assert shuffled > both
    assert report["deltas"]["w_kill_test_passed"] is True
    by_field = {r["field"]: r for r in report["w_ablation"]}
    assert by_field["elevation"]["earns_keep"] is True
    assert "nearest_fire_perimeter_distance_m" not in by_field
    assert "elevation" in report["w_ablation_keep"]
