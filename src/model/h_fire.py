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


def model_infer(
    site_id: str,
    w_features: WFeatures,
    e_features: EFeatures,
    vintages: dict[str, Any] | None = None,
) -> ModelOutput:
    """{site_id, X|null, W, E, vintages} -> {y_hat, sigma, model_version, baseline_y}.

    V1 has no X (EO embedding); the contract accepts it as `vintages`-adjacent metadata
    only, never as a live image (SRS 6.3 / 2.6.3).
    """
    baseline_y = compute_baseline(e_features)
    artefact = _REGISTRY.get()

    if artefact is None:
        return ModelOutput(y_hat=0.1, sigma=0.5, baseline_y=baseline_y, model_version=DUMMY_MODEL_VERSION)

    pipeline = artefact["pipeline"]
    model_version = artefact.get("model_version", "h_fire_unknown")

    e_vec = e_features_to_vector(e_features)
    masked_w = w_features.vector * w_features.mask
    feature_vec = np.concatenate([masked_w, e_vec]).reshape(1, -1)

    try:
        proba = pipeline.predict_proba(feature_vec)[0]
        y_hat = float(proba[1]) if len(proba) > 1 else float(proba[0])
    except Exception as exc:  # a shape/version mismatch must degrade, never crash the agent
        logger.warning("model_infer prediction failed for %s: %s; falling back to baseline", site_id, exc)
        return ModelOutput(y_hat=baseline_y, sigma=0.5, baseline_y=baseline_y, model_version=model_version)

    sigma = artefact.get("sigma_estimate")
    if sigma is None:
        # Ensemble-free calibrated models: use distance from 0.5 as an inverse confidence
        # proxy, clipped to a sane uncertainty range, rather than fabricating a sigma.
        sigma = max(0.1, 0.5 - abs(y_hat - 0.5))
        logger.warning("model_infer: no persisted sigma_estimate for %s; using proxy sigma", model_version)

    return ModelOutput(y_hat=y_hat, sigma=float(sigma), baseline_y=baseline_y, model_version=model_version)
