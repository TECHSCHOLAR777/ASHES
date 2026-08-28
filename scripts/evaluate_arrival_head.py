"""R2 + R4: full-W GBM calibration of the R1 field on R0 y_72 labels.

Uses every Mireye role A-I. Drops only the geometric E leaks (dist_perim_m,
wind_ros_ellipse_dist_m). Reports per-role retrain ablation, grouped permutation
importance per field, engine-only vs engine+W vs shuffled-W.

    python scripts/evaluate_arrival_head.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.e_packer import E_VECTOR_FIELD_ORDER
from src.features.w_encoder import load_field_catalog, model_feature_layout
from src.model.train import RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.arrival_head")

DEFAULT_INPUT = Path("data/training/real_conus_2015_2023_arrival_hrrr.jsonl")
DEFAULT_OUTPUT = Path("data/models/arrival_head_report.json")

GEOM_E = {"dist_perim_m", "wind_ros_ellipse_dist_m"}
SPREAD_NAMES = ["eta_hours", "eta_sigma_hours", "p_burn_24", "p_burn_48", "p_burn_72"]


def _e_keep_indices() -> list[int]:
    return [i for i, name in enumerate(E_VECTOR_FIELD_ORDER) if name not in GEOM_E]


def load_evaluable(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rec = json.loads(line)
            arr = rec.get("arrival") or {}
            if not arr.get("evaluable_72"):
                continue
            if rec.get("y_72", arr.get("y_72")) is None and arr.get("y_72") is None:
                continue
            if not rec.get("spread_vector_hrrr"):
                continue
            rec["_y"] = int(arr["y_72"])
            rows.append(rec)
    return rows


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    layout = model_feature_layout(load_field_catalog())
    w_names = [name for row in layout for name in row["columns"]]
    e_idx = _e_keep_indices()
    e_names = [E_VECTOR_FIELD_ORDER[i] for i in e_idx]
    names = w_names + e_names + SPREAD_NAMES
    x_rows = []
    y = []
    groups = []
    for rec in rows:
        w = np.array(rec["w_vector"], dtype=np.float64) * np.array(rec["w_mask"], dtype=np.float64)
        e = np.array(rec["e_vector"], dtype=np.float64)[e_idx]
        s = np.array(rec["spread_vector_hrrr"], dtype=np.float64)
        x_rows.append(np.concatenate([w, e, s]))
        y.append(rec["_y"])
        groups.append(rec["event_id"])
    return np.vstack(x_rows), np.array(y), np.array(groups), names


def _split(X, y, groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _gbm():
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE)


def _ap(model, X, y) -> float:
    return float(average_precision_score(y, model.predict_proba(X)[:, 1]))


def _fit_ap(Xtr, ytr, Xte, yte) -> tuple[float, float, object]:
    model = _gbm()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return float(average_precision_score(yte, proba)), float(brier_score_loss(yte, proba)), model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_evaluable(args.input)
    if len(rows) < 20:
        raise SystemExit(f"Need >=20 evaluable_72 rows with HRRR spread; found {len(rows)} in {args.input}")

    X, y, groups, names = _matrix(rows)
    layout = model_feature_layout(load_field_catalog())
    w_dim = layout[-1]["end"]
    e_dim = len(_e_keep_indices())
    Xtr, Xte, ytr, yte = _split(X, y, groups)
    report: dict = {
        "n_evaluable": len(rows),
        "n_events": int(len(set(groups))),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "pos_rate": float(y.mean()),
        "test_pos_rate": float(yte.mean()),
        "geometry_e_dropped": sorted(GEOM_E),
        "w_dim": w_dim,
        "e_dim": e_dim,
    }

    # Engine-only: last 5 columns
    ap_eng, brier_eng, _ = _fit_ap(Xtr[:, -5:], ytr, Xte[:, -5:], yte)
    raw_p72 = Xte[:, -1]
    ap_raw = float(average_precision_score(yte, raw_p72))
    report["engine_p72_rank_ap"] = ap_raw
    report["gbm_engine_only"] = {"pr_auc": ap_eng, "brier": brier_eng}

    ap_w, brier_w, _ = _fit_ap(Xtr[:, :w_dim], ytr, Xte[:, :w_dim], yte)
    report["gbm_w_only"] = {"pr_auc": ap_w, "brier": brier_w}

    ap_e, brier_e, _ = _fit_ap(Xtr[:, w_dim : w_dim + e_dim], ytr, Xte[:, w_dim : w_dim + e_dim], yte)
    report["gbm_e_nogeom"] = {"pr_auc": ap_e, "brier": brier_e}

    ap_full, brier_full, model = _fit_ap(Xtr, ytr, Xte, yte)
    report["gbm_full_w_e_engine"] = {"pr_auc": ap_full, "brier": brier_full}

    rng = np.random.default_rng(RANDOM_STATE)
    Xtr_sw, Xte_sw = Xtr.copy(), Xte.copy()
    Xtr_sw[:, :w_dim] = Xtr[rng.permutation(len(Xtr)), :w_dim]
    Xte_sw[:, :w_dim] = Xte[rng.permutation(len(Xte)), :w_dim]
    ap_sw, brier_sw, _ = _fit_ap(Xtr_sw, ytr, Xte_sw, yte)
    report["gbm_shuffled_w"] = {"pr_auc": ap_sw, "brier": brier_sw}
    report["random_w_delta_ap"] = ap_full - ap_sw
    report["beats_raw_engine"] = ap_full > ap_raw + 0.01
    report["w_kill_test_passed"] = (ap_full > ap_raw + 0.01) and (ap_full - ap_sw > 0.01)

    # Per-role retrain ablation (full W, zero that role).
    role_spans: dict[str, tuple[int, int]] = {}
    for row in layout:
        start, end = role_spans.get(row["role"], (row["start"], row["end"]))
        role_spans[row["role"]] = (min(start, row["start"]), max(end, row["end"]))
    roles = {}
    for role, (start, end) in sorted(role_spans.items()):
        Xtr_a, Xte_a = Xtr.copy(), Xte.copy()
        Xtr_a[:, start:end] = 0.0
        Xte_a[:, start:end] = 0.0
        ap_r, _, _ = _fit_ap(Xtr_a, ytr, Xte_a, yte)
        roles[role] = {
            "columns": [start, end],
            "n_columns": end - start,
            "pr_auc_without_role": ap_r,
            "delta_ap": ap_full - ap_r,
        }
        logger.info("role %s delta_ap=%.4f (without=%.4f)", role, ap_full - ap_r, ap_r)
    report["role_retrain_ablation"] = roles

    # Grouped permutation importance: field groups + spread + remaining E.
    groups_idx: dict[str, list[int]] = {}
    for row in layout:
        groups_idx[f"W:{row['role']}:{row['field']}"] = list(range(row["start"], row["end"]))
    for j, name in enumerate(E_VECTOR_FIELD_ORDER):
        if name in GEOM_E:
            continue
        col = w_dim + _e_keep_indices().index(E_VECTOR_FIELD_ORDER.index(name))
        groups_idx[f"E:{name}"] = [col]
    for k, name in enumerate(SPREAD_NAMES):
        groups_idx[f"ENG:{name}"] = [w_dim + e_dim + k]

    # sklearn permutation_importance is per-column; aggregate max/mean per group after.
    logger.info("permutation importance on %d test rows, %d cols", len(yte), Xte.shape[1])
    perm = permutation_importance(
        model, Xte, yte, scoring="average_precision", n_repeats=10, random_state=RANDOM_STATE, n_jobs=1
    )
    field_imp = []
    for label, cols in groups_idx.items():
        mean = float(np.sum(perm.importances_mean[cols]))
        std = float(np.sqrt(np.sum(np.square(perm.importances_std[cols]))))
        field_imp.append({"group": label, "n_columns": len(cols), "delta_ap": mean, "std": std})
    field_imp.sort(key=lambda row: abs(row["delta_ap"]), reverse=True)
    report["grouped_permutation_importance"] = field_imp
    report["top_15"] = field_imp[:15]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s full_ap=%.4f raw_p72=%.4f shuffleW=%.4f kill=%s", args.output, ap_full, ap_raw, ap_sw, report["w_kill_test_passed"])
    print(json.dumps({k: report[k] for k in report if k != "grouped_permutation_importance"}, indent=2))


if __name__ == "__main__":
    main()
