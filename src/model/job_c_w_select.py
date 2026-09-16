"""Within-fire residual correlation map for W_allowed (not a ranking GBM).

Job C's head stays a logistic calibrator. This module only *selects* a
smaller W subset: site-level Spearman of each allowed field vs the
leave-one-event-out engine residual. The p-value shuffles the residual
*within each fire* so between-fire climate cannot fake a parcel effect.

``_conf`` coverage flags are ranked but never selected. Leaks / t0-fuel
cannot enter — they are already absent from W_allowed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.model.job_c_eval import _engine_matrix, _w_matrix, logo_predict
from src.model.w_allowed import allowed_column_indices

N_PERM = 2000
FDR_Q = 0.10
MIN_ABS_RHO = 0.05
EXPLORATORY_K = 8


def _is_conf(col: str) -> bool:
    return col.endswith("_conf")


def _bh_q(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    q = np.ones(n, dtype=np.float64)
    if n == 0:
        return q
    order = np.argsort(p)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = int(order[rank - 1])
        val = min(prev, p[i] * n / rank)
        q[i] = val
        prev = val
    return np.clip(q, 0.0, 1.0)


def _event_means(x: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    events, inv = np.unique(groups, return_inverse=True)
    means = np.zeros(len(events), dtype=np.float64)
    for i in range(len(events)):
        means[i] = float(np.nanmean(x[inv == i]))
    return events, means


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return 0.0
    rho, _ = spearmanr(a, b, nan_policy="omit")
    return 0.0 if not np.isfinite(rho) else float(rho)


def _group_indices(groups: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(groups == g) for g in np.unique(groups)]


def _pearson_ranks(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if den < 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def _within_event_perm_p(
    w_rank: np.ndarray,
    resid_rank: np.ndarray,
    group_idx: list[np.ndarray],
    observed: float,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> float:
    if len(w_rank) < 16 or len(group_idx) < 4:
        return 1.0
    hits = 0
    abs_obs = abs(observed)
    shuffled = resid_rank.copy()
    for _ in range(n_perm):
        for idx in group_idx:
            if len(idx) < 2:
                continue
            shuffled[idx] = rng.permutation(resid_rank[idx])
        rho = _pearson_ranks(w_rank, shuffled)
        if abs(rho) >= abs_obs - 1e-15:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def correlation_map(
    rows: list[dict],
    include_conf: bool = True,
    n_perm: int = N_PERM,
) -> dict[str, Any]:
    """LOGO engine residual vs each W_allowed column, then roll up to fields."""
    indices, w_names, kept = allowed_column_indices()
    y = np.array([rec["_y"] for rec in rows], dtype=np.float64)
    groups = np.array([rec["event_id"] for rec in rows])
    X_eng = _engine_matrix(rows)
    X_w = _w_matrix(rows, indices)
    p_eng = logo_predict(X_eng, y.astype(int), groups, kind="logistic")
    resid = y - p_eng
    eta = X_eng[:, 0]
    group_idx = _group_indices(groups)
    resid_rank = np.argsort(np.argsort(resid)).astype(np.float64)
    rng = np.random.default_rng(42)
    columns: list[dict[str, Any]] = []
    field_cols: dict[str, list[int]] = {row["field"]: [] for row in kept}
    name_to_i = {n: i for i, n in enumerate(w_names)}
    for row in kept:
        for col in row["columns"]:
            field_cols[row["field"]].append(name_to_i[col])

    for i, name in enumerate(w_names):
        w = X_w[:, i]
        rho_resid = _spearman(w, resid)
        rho_y = _spearman(w, y)
        rho_eta = _spearman(w, eta)
        _, w_e = _event_means(w, groups)
        _, resid_e = _event_means(resid, groups)
        rho_between = _spearman(w_e, resid_e)
        columns.append(
            {
                "column": name,
                "rho_residual": rho_resid,
                "rho_between_event": rho_between,
                "rho_y": rho_y,
                "rho_eta": rho_eta,
                "p_residual": None,
                "is_conf": _is_conf(name),
                "_i": i,
            }
        )

    fields: list[dict[str, Any]] = []
    for row in kept:
        idxs = field_cols[row["field"]]
        value_idxs = [j for j in idxs if not _is_conf(w_names[j])]
        use = value_idxs or idxs
        if row["field"] in w_names:
            use = [name_to_i[row["field"]]]
        best = max((columns[j] for j in use), key=lambda c: abs(c["rho_residual"]))
        w_rank = np.argsort(np.argsort(X_w[:, best["_i"]])).astype(np.float64)
        p_resid = _within_event_perm_p(
            w_rank, resid_rank, group_idx, best["rho_residual"], rng, n_perm=n_perm
        )
        fields.append(
            {
                "field": row["field"],
                "n_columns": len(idxs),
                "value_column": best["column"],
                "rho_residual": best["rho_residual"],
                "rho_between_event": best["rho_between_event"],
                "rho_y": best["rho_y"],
                "rho_eta": best["rho_eta"],
                "p_residual": p_resid,
            }
        )
    field_q = _bh_q(np.array([f["p_residual"] for f in fields]))
    for f, q in zip(fields, field_q):
        f["q_residual"] = float(q)
        f["selected_fdr"] = bool(
            (not str(f["value_column"]).endswith("_conf"))
            and f["q_residual"] <= FDR_Q
            and abs(f["rho_residual"]) >= MIN_ABS_RHO
        )
    for col in columns:
        col.pop("_i", None)
        col["p_residual"] = next(
            (f["p_residual"] for f in fields if f["value_column"] == col["column"]),
            None,
        )

    fields_sorted = sorted(fields, key=lambda f: abs(f["rho_residual"]), reverse=True)
    selected = [f["field"] for f in fields_sorted if f["selected_fdr"]]
    exploratory = [
        f["field"] for f in fields_sorted if not str(f["value_column"]).endswith("_conf")
    ][:EXPLORATORY_K]
    head_fields = selected if selected else exploratory
    return {
        "n_rows": int(len(rows)),
        "n_events": int(len(set(groups.tolist()))),
        "unit": "site Spearman vs LOGO engine residual; p-values shuffle residual within fire",
        "n_perm": n_perm,
        "fdr_q": FDR_Q,
        "min_abs_rho": MIN_ABS_RHO,
        "not_a_218d_gbm": True,
        "columns": columns if include_conf else [c for c in columns if not c["is_conf"]],
        "fields": fields_sorted,
        "selected_fdr_fields": selected,
        "exploratory_top_k": exploratory,
        "head_fields": head_fields,
        "head_rule": (
            "FDR survivors of within-fire residual Spearman (q<=0.10 and |rho|>=0.05) "
            "if any; else exploratory top-8 by |rho_residual|"
        ),
        "_resid": resid,
    }


def gbm_field_ranks(
    rows: list[dict],
    max_fields: int = 8,
    resid: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """In-sample GBM on the engine residual — ranking diagnostic only, not the claim head."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    indices, w_names, kept = allowed_column_indices()
    y = np.array([rec["_y"] for rec in rows], dtype=np.float64)
    groups = np.array([rec["event_id"] for rec in rows])
    X_eng = _engine_matrix(rows)
    X_w = _w_matrix(rows, indices)
    if resid is None:
        p_eng = logo_predict(X_eng, y.astype(int), groups, kind="logistic")
        resid = y - p_eng
    keep_i = [i for i, n in enumerate(w_names) if not _is_conf(n)]
    X = X_w[:, keep_i]
    gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=80, learning_rate=0.08, random_state=42)
    gbm.fit(X, resid)
    perm = permutation_importance(gbm, X, resid, n_repeats=8, random_state=42, n_jobs=1)
    col_imp = {w_names[i]: float(perm.importances_mean[k]) for k, i in enumerate(keep_i)}
    field_imp: dict[str, float] = {}
    for row in kept:
        field_imp[row["field"]] = float(
            sum(col_imp.get(c, 0.0) for c in row["columns"] if not _is_conf(c))
        )
    ranked = sorted(field_imp.items(), key=lambda kv: kv[1], reverse=True)
    return [{"field": f, "gbm_perm_importance": imp} for f, imp in ranked[:max_fields]]
