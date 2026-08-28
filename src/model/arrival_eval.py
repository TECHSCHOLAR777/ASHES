"""R2/R4: full-W GBM head on timed-perimeter y_72 with HRRR-seeded engine features.

Protocol
--------
- X is the entire encoded Mireye W (roles A–I, 200-D including
  ``nearest_fire_perimeter_distance_m``) plus live E *without* the two geometric
  leaks (``dist_perim_m``, ``wind_ros_ellipse_dist_m``) plus the five R1 engine
  floats. Geometry is dropped from the matrix so a GBM cannot replay the V1
  scar-distance ranking; it stays in W so importance can still flag a leak.
- Primary metric: leave-one-event-out pooled PR-AUC (small N; a single 75/25
  split is too noisy with ~30 events).
- Secondary: 20× GroupShuffleSplit at the same test_size/seed as ``train.py``.
- Kill test: full GBM beats raw engine p72 rank by >0.01 **and** beating a
  row-shuffled W by >0.01.
- Importance: retrain-ablate every role and every W/E/ENG field under LOGO;
  grouped permutation (block-shuffle columns of one field) on a held-out GSS
  test set.

The engine behind the five floats is whatever ``spread_run`` served (currently
``rothermel_huygens_v1``), not a compiled ELMFIRE binary.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from src.features.e_packer import E_VECTOR_FIELD_ORDER
from src.features.w_encoder import load_field_catalog, model_feature_layout
from src.model.train import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger("fire_copilot.arrival_eval")

GEOM_E = ("dist_perim_m", "wind_ros_ellipse_dist_m")
SPREAD_NAMES = ("eta_hours", "eta_sigma_hours", "p_burn_24", "p_burn_48", "p_burn_72")
KILL_DELTA = 0.01
GSS_REPEATS = 20
PERM_REPEATS = 30


def e_keep_indices() -> list[int]:
    return [i for i, name in enumerate(E_VECTOR_FIELD_ORDER) if name not in GEOM_E]


def e_keep_names() -> list[str]:
    return [E_VECTOR_FIELD_ORDER[i] for i in e_keep_indices()]


def w_layout() -> list[dict[str, Any]]:
    return model_feature_layout(load_field_catalog())


def make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE)


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores)
    mask = np.isfinite(scores)
    y, scores = y[mask], scores[mask]
    if y.size == 0 or y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, scores))


def brier(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores)
    mask = np.isfinite(scores)
    return float(brier_score_loss(y[mask], scores[mask]))


def load_evaluable(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            rec = json_loads_line(line)
            if rec is None:
                continue
            arr = rec.get("arrival") or {}
            if not arr.get("evaluable_72"):
                continue
            if arr.get("y_72") is None:
                continue
            if not rec.get("spread_vector_hrrr"):
                continue
            rec["_y"] = int(arr["y_72"])
            rows.append(rec)
    return rows


def json_loads_line(line: str) -> dict | None:
    import json

    line = line.strip()
    if not line:
        return None
    return json.loads(line)


def corpus_summary(path: Path) -> dict[str, Any]:
    """R0/R1 bookkeeping over every row, not just the evaluable matrix."""
    import json

    n = 0
    events: set[str] = set()
    src: Counter[str] = Counter()
    n_times_event: dict[str, int] = {}
    methods: Counter[str] = Counter()
    already = eval72 = pos72 = neg72 = 0
    y24 = Counter()
    y48 = Counter()
    y72 = Counter()
    wx: Counter[str] = Counter()
    with_field = 0
    eval_field = 0
    eval_field_pos = 0
    events_eval: set[str] = set()
    events_pos: set[str] = set()
    events_field: set[str] = set()
    for line in path.open():
        rec = json.loads(line)
        n += 1
        eid = rec["event_id"]
        events.add(eid)
        arr = rec.get("arrival") or {}
        src[arr.get("source") or "missing"] += 1
        n_times_event[eid] = max(n_times_event.get(eid, 0), int(arr.get("n_times") or 0))
        for method in arr.get("map_methods") or []:
            methods[method] += 1
        if arr.get("already_burned_at_seed"):
            already += 1
        y24[arr.get("y_24")] += 1
        y48[arr.get("y_48")] += 1
        y72[arr.get("y_72")] += 1
        if arr.get("evaluable_72"):
            eval72 += 1
            events_eval.add(eid)
            if arr.get("y_72") == 1:
                pos72 += 1
                events_pos.add(eid)
            elif arr.get("y_72") == 0:
                neg72 += 1
        wx[rec.get("weather_source") or "missing"] += 1
        if rec.get("spread_vector_hrrr"):
            with_field += 1
            events_field.add(eid)
            if arr.get("evaluable_72"):
                eval_field += 1
                if arr.get("y_72") == 1:
                    eval_field_pos += 1
    nt_hist = Counter(n_times_event.values())
    return {
        "path": str(path),
        "n_rows": n,
        "n_events": len(events),
        "sources": dict(src),
        "event_n_times": dict(sorted(nt_hist.items())),
        "events_n_times_ge2": int(sum(1 for v in n_times_event.values() if v >= 2)),
        "already_burned_at_seed": already,
        "evaluable_72": eval72,
        "evaluable_72_events": len(events_eval),
        "y_72_pos": pos72,
        "y_72_neg": neg72,
        "y_72_pos_events": len(events_pos),
        "y_24": {str(k): v for k, v in y24.items()},
        "y_48": {str(k): v for k, v in y48.items()},
        "y_72": {str(k): v for k, v in y72.items()},
        "map_methods": methods.most_common(),
        "weather_source": dict(wx),
        "rows_with_hrrr_field": with_field,
        "events_with_hrrr_field": len(events_field),
        "evaluable_72_with_field": eval_field,
        "evaluable_72_with_field_pos": eval_field_pos,
    }


def build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    layout = w_layout()
    w_names = [name for row in layout for name in row["columns"]]
    e_idx = e_keep_indices()
    e_names = [E_VECTOR_FIELD_ORDER[i] for i in e_idx]
    names = w_names + e_names + list(SPREAD_NAMES)
    x_rows = []
    y = []
    groups = []
    leak_dist = []
    leak_ellipse = []
    e_index = {name: i for i, name in enumerate(E_VECTOR_FIELD_ORDER)}
    for rec in rows:
        w = np.array(rec["w_vector"], dtype=np.float64) * np.array(rec["w_mask"], dtype=np.float64)
        if w.shape[0] != len(w_names):
            raise ValueError(f"w_vector length {w.shape[0]} != layout {len(w_names)}")
        e_full = np.array(rec["e_vector"], dtype=np.float64)
        e = e_full[e_idx]
        s = np.array(rec["spread_vector_hrrr"], dtype=np.float64)
        x_rows.append(np.concatenate([w, e, s]))
        y.append(rec["_y"])
        groups.append(rec["event_id"])
        leak_dist.append(float(e_full[e_index["dist_perim_m"]]))
        leak_ellipse.append(float(e_full[e_index["wind_ros_ellipse_dist_m"]]))
    meta = {
        "layout": layout,
        "w_dim": len(w_names),
        "e_dim": len(e_names),
        "e_names": e_names,
        "leak_dist_perim_m": np.array(leak_dist, dtype=np.float64),
        "leak_wind_ros_ellipse_dist_m": np.array(leak_ellipse, dtype=np.float64),
        "roles": sorted({row["role"] for row in layout}),
    }
    return np.vstack(x_rows), np.array(y, dtype=np.int64), np.array(groups), names, meta


def _logo_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros(len(groups))
    return list(LeaveOneGroupOut().split(dummy, dummy, groups))


def logo_oos_proba(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    columns: np.ndarray | slice | list[int] | None = None,
) -> np.ndarray:
    """Fit a GBM on each leave-one-event-out train fold; return OOS probabilities."""
    if splits is None:
        splits = _logo_splits(groups)
    Xs = X if columns is None else X[:, columns]
    oos = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, test_idx in splits:
        Xtr, ytr = Xs[train_idx], y[train_idx]
        Xte = Xs[test_idx]
        if len(np.unique(ytr)) < 2:
            oos[test_idx] = float(ytr[0])
            continue
        model = make_gbm()
        model.fit(Xtr, ytr)
        oos[test_idx] = model.predict_proba(Xte)[:, 1]
    return oos


def _score_pack(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {"pr_auc": pr_auc(y, proba), "brier": brier(y, proba)}


def role_spans(layout: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    for row in layout:
        start, end = spans.get(row["role"], (row["start"], row["end"]))
        spans[row["role"]] = (min(start, row["start"]), max(end, row["end"]))
    return spans


def feature_groups(layout: list[dict[str, Any]], e_names: list[str], w_dim: int, e_dim: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for row in layout:
        groups[f"W:{row['role']}:{row['field']}"] = list(range(row["start"], row["end"]))
    for j, name in enumerate(e_names):
        groups[f"E:{name}"] = [w_dim + j]
    for k, name in enumerate(SPREAD_NAMES):
        groups[f"ENG:{name}"] = [w_dim + e_dim + k]
    return groups


def grouped_permutation_importance(
    model: HistGradientBoostingClassifier,
    X: np.ndarray,
    y: np.ndarray,
    groups_idx: dict[str, list[int]],
    n_repeats: int = PERM_REPEATS,
    seed: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Block-shuffle every column of a field together; delta is base AP minus shuffled AP."""
    rng = np.random.default_rng(seed)
    base = pr_auc(y, model.predict_proba(X)[:, 1])
    out: list[dict[str, Any]] = []
    for label, cols in groups_idx.items():
        deltas = np.empty(n_repeats, dtype=np.float64)
        col_idx = np.array(cols, dtype=int)
        for i in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, col_idx] = X[perm][:, col_idx]
            deltas[i] = base - pr_auc(y, model.predict_proba(Xp)[:, 1])
        out.append(
            {
                "group": label,
                "n_columns": len(cols),
                "delta_ap": float(np.nanmean(deltas)),
                "std": float(np.nanstd(deltas)),
                "n_repeats": n_repeats,
            }
        )
    out.sort(key=lambda row: abs(row["delta_ap"]), reverse=True)
    return out


def gss_repeat_metrics(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    slices: dict[str, Any],
    n_splits: int = GSS_REPEATS,
) -> dict[str, Any]:
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    bucket: dict[str, list[float]] = defaultdict(list)
    n_used = 0
    for train_idx, test_idx in splitter.split(X, y, groups):
        ytr, yte = y[train_idx], y[test_idx]
        if len(np.unique(ytr)) < 2 or yte.min() == yte.max():
            continue
        n_used += 1
        for name, cols in slices.items():
            model = make_gbm()
            model.fit(X[train_idx][:, cols], ytr)
            proba = model.predict_proba(X[test_idx][:, cols])[:, 1]
            bucket[name].append(pr_auc(yte, proba))
        engine_p72 = X[test_idx][:, -1]
        bucket["rank_engine_p72"].append(pr_auc(yte, engine_p72))
        bucket["rank_neg_eta"].append(pr_auc(yte, -X[test_idx][:, -5]))
    summary = {"n_splits_requested": n_splits, "n_splits_scored": n_used}
    for name, vals in bucket.items():
        arr = np.array(vals, dtype=np.float64)
        summary[name] = {
            "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
            "std": float(np.nanstd(arr)) if arr.size else float("nan"),
            "values": [float(v) for v in arr],
        }
    return summary


def evaluate_arrival_head(rows: list[dict]) -> dict[str, Any]:
    X, y, groups, names, meta = build_matrix(rows)
    layout = meta["layout"]
    w_dim = meta["w_dim"]
    e_dim = meta["e_dim"]
    e_names = meta["e_names"]
    splits = _logo_splits(groups)
    n_events = int(len(set(groups.tolist())))
    logger.info(
        "matrix n=%d events=%d pos=%d w=%d e=%d eng=5 logo_folds=%d",
        len(y), n_events, int(y.sum()), w_dim, e_dim, len(splits),
    )

    all_cols = np.arange(X.shape[1])
    w_cols = np.arange(w_dim)
    e_cols = np.arange(w_dim, w_dim + e_dim)
    eng_cols = np.arange(w_dim + e_dim, X.shape[1])

    rng = np.random.default_rng(RANDOM_STATE)
    X_sw = X.copy()
    X_sw[:, :w_dim] = X[rng.permutation(len(X)), :w_dim]

    logger.info("LOGO full / engine / W / E / shuffled-W")
    proba_full = logo_oos_proba(X, y, groups, splits)
    proba_eng = logo_oos_proba(X, y, groups, splits, eng_cols)
    proba_w = logo_oos_proba(X, y, groups, splits, w_cols)
    proba_e = logo_oos_proba(X, y, groups, splits, e_cols)
    proba_sw = logo_oos_proba(X_sw, y, groups, splits)

    logo_full = _score_pack(y, proba_full)
    logo_eng = _score_pack(y, proba_eng)
    logo_w = _score_pack(y, proba_w)
    logo_e = _score_pack(y, proba_e)
    logo_sw = _score_pack(y, proba_sw)

    rank_p72 = pr_auc(y, X[:, -1])
    rank_eta = pr_auc(y, -X[:, -5])
    peri_idx = names.index("nearest_fire_perimeter_distance_m")
    rank_w_peri = pr_auc(y, -X[:, peri_idx])
    rank_leak_dist = pr_auc(y, -meta["leak_dist_perim_m"])
    rank_leak_ell = pr_auc(y, -meta["leak_wind_ros_ellipse_dist_m"])

    full_ap = logo_full["pr_auc"]
    kill = (full_ap > rank_p72 + KILL_DELTA) and (full_ap - logo_sw["pr_auc"] > KILL_DELTA)

    # Per-role retrain ablation (zero that role, keep the rest of full X).
    logger.info("LOGO per-role retrain ablation")
    roles: dict[str, Any] = {}
    for role, (start, end) in sorted(role_spans(layout).items()):
        Xa = X.copy()
        Xa[:, start:end] = 0.0
        ap_r = pr_auc(y, logo_oos_proba(Xa, y, groups, splits))
        roles[role] = {
            "columns": [start, end],
            "n_columns": end - start,
            "pr_auc_without_role": ap_r,
            "delta_ap": full_ap - ap_r,
        }
        logger.info("role %s delta_ap=%.4f without=%.4f", role, full_ap - ap_r, ap_r)

    # Per-field retrain ablation across W, kept E, and engine floats.
    logger.info("LOGO per-field retrain ablation (%d groups)", len(feature_groups(layout, e_names, w_dim, e_dim)))
    field_ablation = []
    for label, cols in feature_groups(layout, e_names, w_dim, e_dim).items():
        Xa = X.copy()
        Xa[:, cols] = 0.0
        ap_f = pr_auc(y, logo_oos_proba(Xa, y, groups, splits))
        field_ablation.append(
            {
                "group": label,
                "n_columns": len(cols),
                "pr_auc_without_field": ap_f,
                "delta_ap": full_ap - ap_f,
            }
        )
        logger.info("field %s delta_ap=%.4f without=%.4f", label, full_ap - ap_f, ap_f)
    field_ablation.sort(key=lambda row: abs(row["delta_ap"]), reverse=True)

    slices = {
        "gbm_full": all_cols,
        "gbm_engine_only": eng_cols,
        "gbm_w_only": w_cols,
        "gbm_e_nogeom": e_cols,
        "gbm_shuffled_w": all_cols,  # filled below from X_sw on the same splits
    }
    gss = gss_repeat_metrics(X, y, groups, {k: v for k, v in slices.items() if k != "gbm_shuffled_w"})
    gss_sw = gss_repeat_metrics(X_sw, y, groups, {"gbm_shuffled_w": all_cols})
    gss["gbm_shuffled_w"] = gss_sw.get("gbm_shuffled_w", {"mean": float("nan"), "std": float("nan"), "values": []})

    # Grouped permutation on one event-held-out GSS test set (train.py seed).
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    perm: list[dict[str, Any]] = []
    if len(np.unique(y[train_idx])) >= 2 and y[test_idx].min() != y[test_idx].max():
        model = make_gbm()
        model.fit(X[train_idx], y[train_idx])
        logger.info("grouped permutation on %d test rows", len(test_idx))
        perm = grouped_permutation_importance(
            model, X[test_idx], y[test_idx], feature_groups(layout, e_names, w_dim, e_dim)
        )
    else:
        logger.warning("GSS test split is single-class; skipping permutation importance")

    return {
        "n_evaluable": len(rows),
        "n_events": n_events,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "pos_rate": float(y.mean()),
        "w_dim": w_dim,
        "e_dim": e_dim,
        "eng_dim": 5,
        "roles": meta["roles"],
        "geometry_e_dropped": list(GEOM_E),
        "geometry_e_dropped_reason": (
            "V1 y=in-final-MTBS was almost perfectly ranked by -dist_perim_m; "
            "those two E columns are withheld from X so a GBM cannot replay that scar classifier. "
            "nearest_fire_perimeter_distance_m stays in W so importance can still flag it."
        ),
        "engine_identity": "spread_run / rothermel_huygens_v1 (SPREAD_ENGINE_BIN unset); seed=first operational/IR ring; weather=HRRR-at-seed",
        "protocol": {
            "primary": "leave_one_event_out pooled PR-AUC",
            "secondary": f"{GSS_REPEATS}x GroupShuffleSplit test_size={TEST_SIZE} random_state={RANDOM_STATE}",
            "kill_delta": KILL_DELTA,
            "perm_repeats": PERM_REPEATS,
        },
        "rank_baselines": {
            "engine_p72": rank_p72,
            "engine_neg_eta_hours": rank_eta,
            "w_neg_nearest_fire_perimeter_distance_m": rank_w_peri,
            "e_neg_dist_perim_m_NOT_IN_X": rank_leak_dist,
            "e_neg_wind_ros_ellipse_dist_m_NOT_IN_X": rank_leak_ell,
        },
        "logo": {
            "gbm_full_w_e_engine": logo_full,
            "gbm_engine_only": logo_eng,
            "gbm_w_only": logo_w,
            "gbm_e_nogeom": logo_e,
            "gbm_shuffled_w": logo_sw,
            "random_w_delta_ap": full_ap - logo_sw["pr_auc"],
            "beats_raw_engine": full_ap > rank_p72 + KILL_DELTA,
            "w_kill_test_passed": bool(kill),
        },
        "gss20": gss,
        "role_retrain_ablation_logo": roles,
        "field_retrain_ablation_logo": field_ablation,
        "field_retrain_ablation_logo_top_15": field_ablation[:15],
        "grouped_permutation_importance": perm,
        "grouped_permutation_top_15": perm[:15],
        "gss_split_for_perm": {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_pos": int(y[test_idx].sum()),
            "test_events": int(len(set(groups[test_idx].tolist()))),
        },
        "feature_names": names,
    }
