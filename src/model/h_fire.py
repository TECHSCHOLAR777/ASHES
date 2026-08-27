"""model_infer contract (SRS FR-24/FR-27, Step 8). The only place E, W, and X meet.

Loads the latest saved `h_fire_v*.pkl` artefact from data/models/ at first use. If none
exists yet, falls back to a dummy model that still returns a well-formed output (never
raises) so the agent can run end to end before any training has happened - it logs
`MODEL_NOT_TRAINED` as a warning rather than blocking the pipeline (build rule: no stubs
that break the run, degraded paths only).
"""
from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.features.e_packer import EFeatures, e_features_to_vector
from src.features.w_encoder import WFeatures

logger = logging.getLogger("fire_copilot.model")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
DUMMY_MODEL_VERSION = "h_fire_v0_untrained"

# Baseline "distance + ROS" sigmoid (SRS FR-27): fixed, documented constants, not fit from
# data, since a baseline must not itself require training to exist.
_BASELINE_INTERCEPT = 2.0
_BASELINE_DIST_COEF_PER_KM = -0.15
_BASELINE_ROS_COEF_PER_KM = -0.05


@dataclass
class ModelOutput:
    y_hat: float
    sigma: float
    baseline_y: float
    model_version: str
    eta_hours: float | None = None
    eta_sigma_hours: float | None = None
    p_burn_by_T: dict[str, float | None] | None = None
    spread_field_version: str | None = None


def compute_baseline(e_features: EFeatures) -> float:
    """Distance + ROS baseline the model must beat (SRS FR-27), always present."""
    dist_km = min(e_features.dist_perim_m, 999_999.0) / 1000.0
    ros_km = e_features.wind_ros_ellipse_dist_m / 1000.0
    z = _BASELINE_INTERCEPT + _BASELINE_DIST_COEF_PER_KM * dist_km + _BASELINE_ROS_COEF_PER_KM * ros_km
    return 1.0 / (1.0 + math.exp(-z))


def _latest_model_path() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(MODELS_DIR.glob("h_fire_v*.pkl"))
    return candidates[-1] if candidates else None


class _ModelRegistry:
    """Lazily loads and caches the latest trained model artefact."""

    def __init__(self) -> None:
        self._artefact: dict[str, Any] | None = None
        self._loaded_path: Path | None = None
        self._warned_untrained = False

    def get(self) -> dict[str, Any] | None:
        latest = _latest_model_path()
        if latest is None:
            if not self._warned_untrained:
                logger.warning("MODEL_NOT_TRAINED: no h_fire_v*.pkl found in %s; using dummy model", MODELS_DIR)
                self._warned_untrained = True
            return None
        if latest != self._loaded_path:
            with open(latest, "rb") as f:
                self._artefact = pickle.load(f)
            self._loaded_path = latest
        return self._artefact


_REGISTRY = _ModelRegistry()


def _spread_features(spread: Any) -> np.ndarray:
    """Fixed-order raw-field features the V2 calibration head concatenates onto [g(W); E]."""
    eta = spread.eta_hours if spread.eta_hours is not None else 72.0
    sig = spread.eta_sigma_hours if spread.eta_sigma_hours is not None else 24.0
    p24 = spread.p_burn_24 if spread.p_burn_24 is not None else 0.0
    p48 = spread.p_burn_48 if spread.p_burn_48 is not None else 0.0
    p72 = spread.p_burn_72 if spread.p_burn_72 is not None else 0.0
    return np.array([eta, sig, p24, p48, p72], dtype=np.float64)


def _p_burn_dict(spread: Any | None) -> dict[str, float | None] | None:
    if spread is None:
        return None
    return {"24": spread.p_burn_24, "48": spread.p_burn_48, "72": spread.p_burn_72}


def _sigma_from_eta(eta_sigma_hours: float | None, y_hat: float) -> float:
    """Map arrival-time ensemble sigma (hours) onto the 0-1 ActionCard sigma.

    12 h of ETA sigma (the policy evacuate-suppress cutoff) maps to ~0.35, so the
    existing NFR-17 suppress rule still fires when the field is uncertain. Missing
    ensemble sigma falls back to the V1 distance-from-0.5 proxy.
    """
    if eta_sigma_hours is None:
        return max(0.1, 0.5 - abs(y_hat - 0.5))
    return float(min(0.9, max(0.05, eta_sigma_hours / 12.0 * 0.35)))


def model_infer(
    site_id: str,
    w_features: WFeatures,
    e_features: EFeatures,
    vintages: dict[str, Any] | None = None,
    spread: Any | None = None,
) -> ModelOutput:
    """{site_id, X|null, W, E, spread, vintages} -> {y_hat, sigma, baseline_y, eta_*, p_burn_by_T, versions}.

    Signature is stable: `spread` is optional so V1 callers keep working. V2
    `baseline_y` is the raw delegated p_burn_72 when a field exists (FR-27).
    """
    baseline_y = compute_baseline(e_features)
    if spread is not None and spread.p_burn_72 is not None:
        baseline_y = float(spread.p_burn_72)

    artefact = _REGISTRY.get()
    p_burn = _p_burn_dict(spread)
    eta = spread.eta_hours if spread is not None else None
    eta_sigma = spread.eta_sigma_hours if spread is not None else None
    field_version = spread.spread_field_version if spread is not None else None

    if artefact is None:
        y_hat = float(spread.p_burn_72) if (spread is not None and spread.p_burn_72 is not None) else 0.1
        sigma = _sigma_from_eta(eta_sigma, y_hat) if spread is not None else 0.5
        return ModelOutput(
            y_hat=y_hat,
            sigma=sigma,
            baseline_y=baseline_y,
            model_version=DUMMY_MODEL_VERSION,
            eta_hours=eta,
            eta_sigma_hours=eta_sigma,
            p_burn_by_T=p_burn,
            spread_field_version=field_version,
        )

    pipeline = artefact["pipeline"]
    model_version = artefact.get("model_version", "h_fire_unknown")
    e_vec = e_features_to_vector(e_features)
    masked_w = w_features.vector * w_features.mask

    if artefact.get("v2_calibration") and spread is not None:
        feature_vec = np.concatenate([masked_w, e_vec, _spread_features(spread)]).reshape(1, -1)
    else:
        feature_vec = np.concatenate([masked_w, e_vec]).reshape(1, -1)

    try:
        # V2 calibration is still a probability head (FR-26): it conditions on the
        # raw arrival field, it does not replace the engine's ETA / P(burn by T).
        proba = pipeline.predict_proba(feature_vec)[0]
        y_hat = float(proba[1]) if len(proba) > 1 else float(proba[0])
    except Exception as exc:  # a shape/version mismatch must degrade, never crash the agent
        logger.warning("model_infer prediction failed for %s: %s; falling back to baseline", site_id, exc)
        y_hat = baseline_y
        return ModelOutput(
            y_hat=y_hat,
            sigma=0.5,
            baseline_y=baseline_y,
            model_version=model_version,
            eta_hours=eta,
            eta_sigma_hours=eta_sigma,
            p_burn_by_T=p_burn,
            spread_field_version=field_version,
        )

    sigma = artefact.get("sigma_estimate")
    if sigma is None:
        sigma = _sigma_from_eta(eta_sigma, y_hat) if spread is not None else max(0.1, 0.5 - abs(y_hat - 0.5))
        logger.warning("model_infer: no persisted sigma_estimate for %s; using proxy sigma", model_version)

    return ModelOutput(
        y_hat=y_hat,
        sigma=float(sigma),
        baseline_y=baseline_y,
        model_version=model_version,
        eta_hours=eta,
        eta_sigma_hours=eta_sigma,
        p_burn_by_T=p_burn,
        spread_field_version=field_version,
    )
