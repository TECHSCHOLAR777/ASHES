"""Offline training for h_fire (SRS Step 8 / 6.4 / 6.5). CPU-only, no GPU required.

Trains a small MLPClassifier wrapped in isotonic calibration on an event-held-out split,
runs the mandatory random-W collapse probe (SRS 6.5), and persists the model artefact plus
a metrics JSON. Model training here is deliberately small (hidden layers capped at
128 units, max_iter=500) so this always finishes in seconds on a laptop CPU - the SRS's
own guidance is "V1 tabular model trains on CPU/small GPU" (SRS 2.5); this build never
assumes or requires a GPU (see DECISIONS.md).
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("fire_copilot.train")

TRAINING_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "training"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (128, 64, 32)
MAX_ITER = 500
CALIBRATION_CV = 5
TEST_SIZE = 0.25


@dataclass
class TrainingSample:
    site_id: str
    event_id: str
    t0: str
    w_vector: list[float]
    w_mask: list[float]
    e_vector: list[float]
    y: int
    spread_vector: list[float] | None = None  # [eta, eta_sigma, p24, p48, p72]
    dist_source: str | None = None


def _has_spread(sample: TrainingSample) -> bool:
    return sample.spread_vector is not None and len(sample.spread_vector) >= 5


def load_samples(training_dir: Path | str = TRAINING_DIR) -> list[TrainingSample]:
    """Load jsonl rows. Duplicate site_ids keep the copy that already has spread_vector
    so Path B's sidecar file can sit next to V1's original jsonl without double-counting."""
    training_dir = Path(training_dir)
    by_id: dict[str, TrainingSample] = {}
    order: list[str] = []
    for path in sorted(training_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample = TrainingSample(
                    site_id=record["site_id"],
                    event_id=record["event_id"],
                    t0=record["t0"],
                    w_vector=record["w_vector"],
                    w_mask=record["w_mask"],
                    e_vector=record["e_vector"],
                    y=int(record["y"]),
                    spread_vector=record.get("spread_vector"),
                    dist_source=record.get("dist_source"),
                )
                prev = by_id.get(sample.site_id)
                if prev is None:
                    by_id[sample.site_id] = sample
                    order.append(sample.site_id)
                elif _has_spread(sample) and not _has_spread(prev):
                    by_id[sample.site_id] = sample
    return [by_id[site_id] for site_id in order]


def _build_feature_matrix(
    samples: list[TrainingSample], include_spread: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for s in samples:
        parts = [np.array(s.w_vector) * np.array(s.w_mask), np.array(s.e_vector)]
        if include_spread:
            parts.append(np.array(s.spread_vector, dtype=np.float64))
        rows.append(np.concatenate(parts))
    X = np.array(rows)
    y = np.array([s.y for s in samples])
    groups = np.array([s.event_id for s in samples])
    return X, y, groups


def _make_pipeline() -> Pipeline:
    mlp = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER, random_state=RANDOM_STATE
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("calibrated", CalibratedClassifierCV(mlp, method="isotonic", cv=CALIBRATION_CV)),
        ]
    )


def _event_held_out_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits by event_id (SRS 6.5): no fire's samples appear in both train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _random_w_collapse_probe(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, w_dim: int
) -> float:
    """Shuffles the W columns and re-evaluates; a real skill gain should collapse (SRS 6.5)."""
    rng = np.random.default_rng(RANDOM_STATE)
    X_train_shuffled = X_train.copy()
    X_test_shuffled = X_test.copy()
    perm = rng.permutation(X_train_shuffled.shape[0])
    X_train_shuffled[:, :w_dim] = X_train_shuffled[perm][:, :w_dim]
    perm_test = rng.permutation(X_test_shuffled.shape[0])
    X_test_shuffled[:, :w_dim] = X_test_shuffled[perm_test][:, :w_dim]

    pipeline = _make_pipeline()
    pipeline.fit(X_train_shuffled, y_train)
    proba = pipeline.predict_proba(X_test_shuffled)[:, 1]
    return float(average_precision_score(y_test, proba))


def train_model(training_dir: str | Path = TRAINING_DIR) -> dict[str, Any]:
    samples = load_samples(training_dir)
    if len(samples) < 10:
        raise ValueError(f"Not enough training samples in {training_dir} (found {len(samples)}, need >= 10)")

    v2_samples = [
        s for s in samples if s.spread_vector is not None and len(s.spread_vector) >= 5
    ]
    if len(v2_samples) >= 10:
        return _train_v2_calibration(v2_samples)

    X, y, groups = _build_feature_matrix(samples)
    w_dim = len(samples[0].w_vector)

    X_train, X_test, y_train, y_test = _event_held_out_split(X, y, groups)

    pipeline = _make_pipeline()
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba_test))
    brier = float(brier_score_loss(y_test, proba_test))

    ap_shuffled_w = _random_w_collapse_probe(X_train, y_train, X_test, y_test, w_dim)
    random_w_delta_ap = ap - ap_shuffled_w

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    model_version = f"h_fire_v{timestamp}"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "pipeline": pipeline,
                "model_version": model_version,
                "sigma_estimate": None,  # V1: h_fire.py falls back to a proxy sigma
                "w_dim": w_dim,
                "trained_at": timestamp,
            },
            f,
        )

    metrics = {
        "model_version": model_version,
        "n_samples": len(samples),
        "n_events": int(len(set(groups))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pr_auc": ap,
        "brier_score": brier,
        "random_w_shuffled_pr_auc": ap_shuffled_w,
        "random_w_collapse_delta_ap": random_w_delta_ap,
        "h1_signal": random_w_delta_ap > 0.01,
    }
    metrics_path = MODELS_DIR / f"metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Trained %s: PR-AUC=%.4f Brier=%.4f random-W delta AP=%.4f", model_version, ap, brier, random_w_delta_ap)
    return metrics


def _train_v2_calibration(samples: list[TrainingSample], persist: bool = True) -> dict[str, Any]:
    """W-conditioned calibration head on the raw delegated field (FR-26 / SRS 6.6).

    Features are [g(W); E; raw_eta, raw_sigma, p24, p48, p72]. Target is the V1
    gold label (in final perimeter). ETA itself stays the engine's ensemble mean;
    this head calibrates P(burn), it does not relearn Rothermel.
    """
    X, y, groups = _build_feature_matrix(samples, include_spread=True)
    w_dim = len(samples[0].w_vector)
    raw_p72 = np.array([float(s.spread_vector[4]) for s in samples], dtype=np.float64)

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    raw_test = raw_p72[test_idx]

    pipeline = _make_pipeline()
    pipeline.fit(X_train, y_train)
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba_test))
    brier = float(brier_score_loss(y_test, proba_test))
    ap_raw = float(average_precision_score(y_test, raw_test))
    brier_raw = float(brier_score_loss(y_test, np.clip(raw_test, 0.0, 1.0)))

    ap_shuffled_w = _random_w_collapse_probe(X_train, y_train, X_test, y_test, w_dim)
    random_w_delta_ap = ap - ap_shuffled_w
    beats_raw = ap > ap_raw + 0.01
    h2_validated = bool(beats_raw and random_w_delta_ap > 0.01)
    h2_falsified = bool((not beats_raw) and random_w_delta_ap <= 0.01)
    if h2_validated:
        h2_status = "validated"
    elif h2_falsified:
        h2_status = "falsified"
    else:
        h2_status = "inconclusive"

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    model_version = f"h_fire_v{timestamp}"
    metrics = {
        "model_version": model_version,
        "v2_calibration": True,
        "n_samples": len(samples),
        "n_events": int(len(set(groups))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pr_auc": ap,
        "brier_score": brier,
        "raw_engine_pr_auc": ap_raw,
        "raw_engine_brier": brier_raw,
        "calibrated_minus_raw_ap": ap - ap_raw,
        "random_w_shuffled_pr_auc": ap_shuffled_w,
        "random_w_collapse_delta_ap": random_w_delta_ap,
        "h2_beats_raw_elmfire": beats_raw,
        "h2_status": h2_status,
        "h2_validated": h2_validated,
        "h2_falsified": h2_falsified,
    }
    if persist:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "pipeline": pipeline,
                    "model_version": model_version,
                    "sigma_estimate": None,
                    "w_dim": w_dim,
                    "spread_dim": 5,
                    "v2_calibration": True,
                    "trained_at": timestamp,
                },
                f,
            )
        metrics_path = MODELS_DIR / f"metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    logger.info(
        "Trained V2 %s: PR-AUC=%.4f raw=%.4f random-W delta=%.4f h2=%s persist=%s",
        model_version, ap, ap_raw, random_w_delta_ap, h2_status, persist,
    )
    return metrics


def synthetic_h2_probe(n_samples: int = 400, n_events: int = 40, seed: int = RANDOM_STATE) -> dict[str, Any]:
    """In-memory H2 machinery check. Labels are constructed so W actually helps.

    This is NOT a research claim. Real H2 needs event/HUC/state held-out fires
    with a real delegated field. See evaluate_h2_claim.
    """
    rng = np.random.default_rng(seed)
    w_dim = 16
    e_dim = 15
    samples: list[TrainingSample] = []
    for i in range(n_samples):
        fuel = float(rng.uniform(0.0, 1.0))
        w = rng.normal(0.0, 0.3, size=w_dim)
        w[0] = fuel
        mask = np.ones(w_dim)
        e = rng.normal(0.0, 1.0, size=e_dim)
        raw_p72 = float(np.clip(0.2 + 0.5 * fuel + rng.normal(0.0, 0.15), 0.0, 1.0))
        # Gold: W (fuel) plus raw field, so calibration can beat raw-only.
        p_gold = float(np.clip(0.15 + 0.55 * fuel + 0.3 * raw_p72, 0.0, 1.0))
        y = int(rng.uniform() < p_gold)
        eta = float(max(0.5, 72.0 * (1.0 - raw_p72) + rng.normal(0.0, 2.0)))
        samples.append(
            TrainingSample(
                site_id=f"h2_site_{i:04d}",
                event_id=f"h2_event_{i % n_events:03d}",
                t0="2020-01-01T12:00:00Z",
                w_vector=w.tolist(),
                w_mask=mask.tolist(),
                e_vector=e.tolist(),
                y=y,
                spread_vector=[eta, 3.0, min(1.0, raw_p72 * 0.6), min(1.0, raw_p72 * 0.85), raw_p72],
                dist_source="synthetic_probe",
            )
        )
    metrics = _train_v2_calibration(samples, persist=False)
    metrics["claim"] = "synthetic_probe_only"
    metrics["note"] = (
        "Synthetic probe that checks the H2 evaluation machinery. "
        "Not an AC-12/AC-13 research result."
    )
    return metrics


def evaluate_h2_claim(
    training_dir: str | Path = TRAINING_DIR,
    run_synthetic_probe: bool = True,
    probe_samples: int = 200,
) -> dict[str, Any]:
    """AC-12/AC-13 gate. Honest unevaluable when real spread-labeled fires are missing."""
    samples = load_samples(training_dir)
    v2_samples = [
        s
        for s in samples
        if s.spread_vector is not None
        and len(s.spread_vector) >= 5
        and (s.dist_source or "") != "synthetic_probe"
    ]
    real_events = {s.event_id for s in v2_samples if not str(s.event_id).startswith("h2_event_")}
    report: dict[str, Any] = {
        "n_jsonl_samples": len(samples),
        "n_spread_labeled_real": len(v2_samples),
        "n_real_events": len(real_events),
    }
    if run_synthetic_probe:
        report["synthetic_probe"] = synthetic_h2_probe(n_samples=probe_samples)
    if len(v2_samples) < 10 or len(real_events) < 8:
        report["h2_status"] = "unevaluable_no_real_spread_labels"
        report["h2_validated"] = None
        report["h2_falsified"] = None
        report["note"] = (
            "No real MTBS/WFIGS event set with delegated spread_vector was present "
            f"(found {len(v2_samples)} spread-labeled samples across {len(real_events)} events). "
            "SRS 6.6 requires event/HUC/state held-out fires. Collect with "
            "scripts/enrich_spread_vectors.py (Path B, no extra Mireye credits) or "
            "scripts/build_training_set.py --with-spread, then rerun this gate. "
            "The synthetic_probe key only verifies that the comparison code runs."
        )
        return report

    trained = _train_v2_calibration(v2_samples)
    report.update(trained)
    report["n_spread_labeled_real"] = len(v2_samples)
    return report
