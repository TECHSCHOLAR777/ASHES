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


def load_samples(training_dir: Path | str = TRAINING_DIR) -> list[TrainingSample]:
    training_dir = Path(training_dir)
    samples: list[TrainingSample] = []
    for path in sorted(training_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                samples.append(
                    TrainingSample(
                        site_id=record["site_id"],
                        event_id=record["event_id"],
                        t0=record["t0"],
                        w_vector=record["w_vector"],
                        w_mask=record["w_mask"],
                        e_vector=record["e_vector"],
                        y=int(record["y"]),
                    )
                )
    return samples


def _build_feature_matrix(samples: list[TrainingSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [np.concatenate([np.array(s.w_vector) * np.array(s.w_mask), np.array(s.e_vector)]) for s in samples]
    )
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
