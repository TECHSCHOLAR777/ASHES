from src.model.train import evaluate_h2_claim, synthetic_h2_probe


def test_evaluate_h2_is_unevaluable_without_real_spread_labels(tmp_path):
    report = evaluate_h2_claim(tmp_path, run_synthetic_probe=False)
    assert report["h2_status"] == "unevaluable_no_real_spread_labels"
    assert report["h2_validated"] is None
    assert report["n_spread_labeled_real"] == 0


def test_synthetic_h2_probe_runs_without_persisting(tmp_path, monkeypatch):
    from src.model import train as train_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path / "models")
    metrics = synthetic_h2_probe(n_samples=80, n_events=16)
    assert metrics["claim"] == "synthetic_probe_only"
    assert "pr_auc" in metrics
    assert "raw_engine_pr_auc" in metrics
    assert metrics["h2_status"] in {"validated", "falsified", "inconclusive"}
    assert not list((tmp_path / "models").glob("h_fire_v*.pkl"))
