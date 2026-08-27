"""Runs the SRS 6.5 mandatory per-role W ablation, plus a gradient-boosted-tree comparison
baseline, on the real training set. Neither needs new Mireye credits - both reuse the
already-collected `data/training/*.jsonl` samples.

    python scripts/ablation_study.py

Per-role ablation: for each Mireye W role (A-I), zero that role's columns out (mask=0, same
mechanism a genuinely missing field already uses) and retrain. A role whose removal barely
changes PR-AUC is not earning its keep in the ~61-field Fire Intelligence Set; a role whose
removal hurts a lot is confirmed load-bearing. This is exactly the counterweight the SRS
calls for against fetching a generous field set (6.5): "per-role W ablation... to confirm
the rich set earns its keep and to prune roles that add nothing."
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features.w_encoder import _category_slots, load_field_catalog, ordered_model_fields  # noqa: E402
from src.model.train import (  # noqa: E402
    RANDOM_STATE,
    TRAINING_DIR,
    _build_feature_matrix,
    _event_held_out_split,
    _make_pipeline,
    load_samples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.ablation")


def role_index_ranges(catalog: dict) -> dict[str, tuple[int, int]]:
    """(start, end) column range per W role in the encoded W vector. `ordered_model_fields`
    sorts by role then field name, so each role's columns are contiguous - this just walks
    the same field list `encode_w` does and counts slot widths per field (value slots + the
    always-appended `_conf` bit) to find the boundaries, without touching `encode_w` itself.
    """
    fields = ordered_model_fields(catalog)
    ranges: dict[str, list[int]] = {}
    idx = 0
    for role, name, meta in fields:
        ftype = meta["type"]
        if ftype == "unordered_categorical":
            n_value = len(_category_slots(meta["categories"]))
        else:
            n_value = 1  # float/int/bool/ordered_categorical, incl. most_recent_burn_year
        width = n_value + 1  # + confidence bit
        if role not in ranges:
            ranges[role] = [idx, idx]
        ranges[role][1] = idx + width
        idx += width
    return {role: (start, end) for role, (start, end) in ranges.items()}


def run_ablation(training_dir: str | Path = TRAINING_DIR) -> dict:
    samples = load_samples(training_dir)
    X, y, groups = _build_feature_matrix(samples)
    w_dim = len(samples[0].w_vector)
    X_train, X_test, y_train, y_test = _event_held_out_split(X, y, groups)

    pipeline = _make_pipeline()
    pipeline.fit(X_train, y_train)
    full_ap = float(average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1]))
    logger.info("Full model (all roles): PR-AUC=%.4f", full_ap)

    catalog = load_field_catalog()
    ranges = role_index_ranges(catalog)

    results = {"full_model_pr_auc": full_ap, "roles": {}}
    for role, (start, end) in sorted(ranges.items()):
        X_train_ablated = X_train.copy()
        X_test_ablated = X_test.copy()
        X_train_ablated[:, start:end] = 0.0
        X_test_ablated[:, start:end] = 0.0

        role_pipeline = _make_pipeline()
        role_pipeline.fit(X_train_ablated, y_train)
        ap = float(average_precision_score(y_test, role_pipeline.predict_proba(X_test_ablated)[:, 1]))
        delta = full_ap - ap
        results["roles"][role] = {
            "columns": [start, end],
            "n_columns": end - start,
            "pr_auc_without_role": ap,
            "delta_ap": delta,
            "earns_its_keep": delta > 0.005,
        }
        logger.info(
            "Role %s (%d cols, idx %d:%d): PR-AUC without it=%.4f, delta=%.4f%s",
            role, end - start, start, end, ap, delta, " [WEAK]" if delta <= 0.005 else "",
        )

    return results


def run_gbm_comparison(training_dir: str | Path = TRAINING_DIR) -> dict:
    samples = load_samples(training_dir)
    X, y, groups = _build_feature_matrix(samples)
    X_train, X_test, y_train, y_test = _event_held_out_split(X, y, groups)

    gbm_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "calibrated",
                CalibratedClassifierCV(
                    HistGradientBoostingClassifier(random_state=RANDOM_STATE), method="isotonic", cv=5
                ),
            ),
        ]
    )
    gbm_pipeline.fit(X_train, y_train)
    proba = gbm_pipeline.predict_proba(X_test)[:, 1]
    ap = float(average_precision_score(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))
    logger.info("HistGradientBoostingClassifier: PR-AUC=%.4f Brier=%.4f", ap, brier)
    return {"model": "HistGradientBoostingClassifier", "pr_auc": ap, "brier_score": brier}


def main() -> None:
    logger.info("=== Per-role W ablation (SRS 6.5) ===")
    ablation_results = run_ablation()

    logger.info("=== Gradient-boosted-tree comparison ===")
    gbm_results = run_gbm_comparison()

    out = {"ablation": ablation_results, "gbm_comparison": gbm_results}
    out_path = Path(__file__).resolve().parent.parent / "data" / "models" / "ablation_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
