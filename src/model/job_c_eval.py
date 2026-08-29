"""Job C calibrator: P(y_72 | engine) vs P(y_72 | engine, W_allowed).

This is not a 218-D ranking GBM. The claim is calibration of a small logistic
head on the ELMFIRE arrival clock, with W_allowed explaining residual hit rate
at the same ETA. Protocol is leave-one-event-out. Metrics are Brier, log-loss,
reliability, and a shuffle-W kill — PR-AUC is a side check only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.model.arrival_eval import pr_auc
from src.model.w_allowed import allowed_column_indices, slice_w

ENGINE_NAMES = ("eta_hours", "eta_sigma_hours", "p_burn_24", "p_burn_48", "p_burn_72")
KILL_BRIER = 0.005
RELIABILITY_BINS = 10
ETA_BINS_H = (0.0, 6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0)


def _finite_clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.isfinite(p)
    if mask.sum() == 0 or y[mask].min() == y[mask].max():
        # sklearn log_loss needs both classes; still report on the finite slice if possible
        if mask.sum() == 0:
            return float("nan")
    return float(log_loss(y[mask], _finite_clip01(p[mask]), labels=[0, 1]))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.isfinite(p)
    return float(brier_score_loss(y[mask], p[mask]))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = RELIABILITY_BINS) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            sel = (p >= lo) & (p <= hi)
        else:
            sel = (p >= lo) & (p < hi)
        count = int(sel.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "pred": None, "obs": None})
            continue
        pred = float(p[sel].mean())
        obs = float(y[sel].mean())
        ece += (count / n) * abs(pred - obs)
        bins.append({"lo": float(lo), "hi": float(hi), "n": count, "pred": pred, "obs": obs})
    return {"n_bins": n_bins, "ece": float(ece), "bins": bins}


def load_evaluable_job_c(path) -> list[dict]:
    import json
    from pathlib import Path

    rows: list[dict] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            arr = rec.get("arrival") or {}
            if not arr.get("evaluable_72") or arr.get("y_72") is None:
                continue
            vec = rec.get("spread_vector_elmfire") or rec.get("spread_vector")
            if not vec:
                continue
            engine = rec.get("engine") or rec.get("spread_engine") or ""
            if "huygens" in str(engine).lower():
                continue
            rec["_y"] = int(arr["y_72"])
            rec["_spread"] = [float(x) if x is not None else float("nan") for x in vec]
            if not rec.get("w_vector") or rec.get("w_mask") is None:
                continue
            rows.append(rec)
    return rows


def _engine_matrix(rows: list[dict]) -> np.ndarray:
    """ELMFIRE clock. Missing ETA means the cell never burned in the 72 h field."""
    X = np.zeros((len(rows), 6), dtype=np.float64)
    for i, rec in enumerate(rows):
        eta, sigma, p24, p48, p72 = rec["_spread"][:5]
        eta = eta if np.isfinite(eta) else 72.0
        sigma = sigma if np.isfinite(sigma) else 0.0
        p24 = p24 if np.isfinite(p24) else 0.0
        p48 = p48 if np.isfinite(p48) else 0.0
        p72 = p72 if np.isfinite(p72) else 0.0
        X[i] = [eta, math_log1p(eta), sigma, p24, p48, p72]
    return X


def math_log1p(x: float) -> float:
    return float(np.log1p(max(0.0, x)))


def _w_matrix(rows: list[dict], indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros((len(rows), 0), dtype=np.float64)
    return np.vstack(
        [slice_w(rec["w_vector"], rec["w_mask"], indices) for rec in rows]
    )


def _fit_logistic(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr[0]))
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    model = LogisticRegression(max_iter=400, solver="lbfgs")
    model.fit(Xtr_s, ytr)
    return model.predict_proba(Xte_s)[:, 1]


def _fit_isotonic_eta(eta_tr: np.ndarray, ytr: np.ndarray, eta_te: np.ndarray) -> np.ndarray:
    if len(np.unique(ytr)) < 2:
        return np.full(len(eta_te), float(ytr[0]))
    # Hit probability should fall as ETA grows.
    iso = IsotonicRegression(out_of_bounds="clip", increasing=False)
    iso.fit(eta_tr, ytr)
    return np.asarray(iso.predict(eta_te), dtype=np.float64)


def logo_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str = "logistic",
) -> np.ndarray:
    dummy = np.zeros(len(groups))
    oos = np.full(len(y), np.nan, dtype=np.float64)
    for train_idx, test_idx in LeaveOneGroupOut().split(dummy, dummy, groups):
        if kind == "isotonic_eta":
            oos[test_idx] = _fit_isotonic_eta(X[train_idx, 0], y[train_idx], X[test_idx, 0])
        else:
            oos[test_idx] = _fit_logistic(X[train_idx], y[train_idx], X[test_idx])
    return oos


def same_eta_slices(
    y: np.ndarray,
    eta: np.ndarray,
    p_engine: np.ndarray,
    p_w: np.ndarray,
    bins: tuple[float, ...] = ETA_BINS_H,
) -> list[dict[str, Any]]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (eta >= lo) & (eta < hi) & np.isfinite(eta)
        n = int(sel.sum())
        if n == 0:
            continue
        ys = y[sel]
        row = {
            "eta_lo": lo,
            "eta_hi": hi,
            "n": n,
            "n_pos": int(ys.sum()),
            "obs_rate": float(ys.mean()),
            "engine_mean_p": float(np.nanmean(p_engine[sel])),
            "engine_w_mean_p": float(np.nanmean(p_w[sel])),
            "engine_brier": brier(ys, p_engine[sel]) if n >= 2 and ys.min() != ys.max() or n >= 2 else float("nan"),
            "engine_w_brier": brier(ys, p_w[sel]) if n >= 2 else float("nan"),
        }
        if n >= 8 and ys.min() != ys.max():
            row["engine_brier"] = brier(ys, p_engine[sel])
            row["engine_w_brier"] = brier(ys, p_w[sel])
            row["delta_brier"] = row["engine_brier"] - row["engine_w_brier"]
        out.append(row)
    return out


def evaluate_job_c(rows: list[dict], include_fields: list[str] | None = None) -> dict[str, Any]:
    if len(rows) < 20:
        raise ValueError(f"Job C needs more evaluable rows than {len(rows)}")
    indices, w_names, kept = allowed_column_indices(include_fields=include_fields)
    y = np.array([rec["_y"] for rec in rows], dtype=np.int64)
    groups = np.array([rec["event_id"] for rec in rows])
    X_eng = _engine_matrix(rows)
    X_w = _w_matrix(rows, indices)
    X_both = np.hstack([X_eng, X_w])
    rng = np.random.default_rng(42)
    X_sw = X_both.copy()
    X_sw[:, X_eng.shape[1] :] = X_w[rng.permutation(len(rows))]

    p_raw72 = X_eng[:, 5]
    p_iso = logo_predict(X_eng, y, groups, kind="isotonic_eta")
    p_eng = logo_predict(X_eng, y, groups, kind="logistic")
    p_both = logo_predict(X_both, y, groups, kind="logistic")
    p_sw = logo_predict(X_sw, y, groups, kind="logistic")

    brier_eng = brier(y, p_eng)
    brier_both = brier(y, p_both)
    brier_sw = brier(y, p_sw)
    brier_iso = brier(y, p_iso)
    brier_raw = brier(y, p_raw72)
    delta = brier_eng - brier_both
    shuffle_delta = brier_sw - brier_both
    kill = bool(delta > KILL_BRIER and shuffle_delta > KILL_BRIER)

    eta = X_eng[:, 0]
    n_events = int(len(set(groups.tolist())))
    n_eta_imputed = int(sum(1 for rec in rows if not np.isfinite(rec["_spread"][0])))
    n_p72_imputed = int(sum(1 for rec in rows if not np.isfinite(rec["_spread"][4])))
    return {
        "n_evaluable": int(len(rows)),
        "n_events": n_events,
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "pos_rate": float(y.mean()),
        "n_eta_imputed_to_72h": n_eta_imputed,
        "n_p72_imputed_to_0": n_p72_imputed,
        "w_allowed_dim": int(X_w.shape[1]),
        "w_allowed_fields": sorted({row["field"] for row in kept}),
        "w_allowed_columns": w_names,
        "w_include_fields": list(include_fields) if include_fields is not None else None,
        "engine_identity_required": "elmfire (SPREAD_ENGINE_REQUIRED; Huygens rows dropped)",
        "protocol": {
            "primary": "leave_one_event_out Brier / log-loss of a logistic calibrator",
            "engine_features": ["eta_hours", "log1p(eta)", "eta_sigma_hours", "p_burn_24", "p_burn_48", "p_burn_72"],
            "head": "LogisticRegression on StandardScaler features; isotonic(eta) as a one-D engine baseline",
            "kill_brier": KILL_BRIER,
            "not_a_218d_gbm": True,
            "w_subset": include_fields is not None,
        },
        "scores": {
            "raw_engine_p72": {"brier": brier_raw, "log_loss": logloss(y, p_raw72), "pr_auc": pr_auc(y, p_raw72)},
            "isotonic_eta": {"brier": brier_iso, "log_loss": logloss(y, p_iso), "pr_auc": pr_auc(y, -eta)},
            "logistic_engine": {"brier": brier_eng, "log_loss": logloss(y, p_eng), "pr_auc": pr_auc(y, p_eng)},
            "logistic_engine_w_allowed": {
                "brier": brier_both,
                "log_loss": logloss(y, p_both),
                "pr_auc": pr_auc(y, p_both),
            },
            "logistic_engine_shuffled_w": {
                "brier": brier_sw,
                "log_loss": logloss(y, p_sw),
                "pr_auc": pr_auc(y, p_sw),
            },
        },
        "deltas": {
            "brier_engine_minus_engine_w": delta,
            "brier_shuffled_minus_engine_w": shuffle_delta,
            "log_loss_engine_minus_engine_w": logloss(y, p_eng) - logloss(y, p_both),
            "w_kill_test_passed": kill,
        },
        "reliability": {
            "logistic_engine": reliability(y, p_eng),
            "logistic_engine_w_allowed": reliability(y, p_both),
            "raw_engine_p72": reliability(y, p_raw72),
        },
        "same_eta_slices": same_eta_slices(y, eta, p_eng, p_both),
        "constant_prevalence_brier": float(y.mean() * (1.0 - y.mean())),
    }
