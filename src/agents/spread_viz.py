"""Downsample a spread field for the map. Never invents ETA or P(burn) values."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.spread.client import SpreadField, SpreadSiteSample


def _finite(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def sample_dict(sample: SpreadSiteSample | None) -> dict[str, Any] | None:
    if sample is None:
        return None
    return {
        "eta_hours": _finite(sample.eta_hours),
        "eta_sigma_hours": _finite(sample.eta_sigma_hours),
        "p_burn_24": _finite(sample.p_burn_24),
        "p_burn_48": _finite(sample.p_burn_48),
        "p_burn_72": _finite(sample.p_burn_72),
        "spread_field_version": sample.spread_field_version,
        "inside_aoi": bool(sample.inside_aoi),
        "raw_eta_hours": _finite(sample.raw_eta_hours),
    }


def downsample_field(field: SpreadField, max_dim: int = 80) -> dict[str, Any]:
    """Quantized arrival / p72 rasters for Leaflet. NaN stays null (transparent)."""
    arrival = np.asarray(field.arrival_hours, dtype=np.float64)
    p72 = np.asarray(field.p_burn_72, dtype=np.float64)
    h, w = arrival.shape
    stride = max(1, int(math.ceil(max(h, w) / max_dim)))
    sub_a = arrival[::stride, ::stride]
    sub_p = p72[::stride, ::stride]
    sh, sw = sub_a.shape

    def _grid(arr: np.ndarray) -> list[list[float | None]]:
        out: list[list[float | None]] = []
        for row in arr:
            out.append([_finite(v) for v in row])
        return out

    finite = sub_a[np.isfinite(sub_a)]
    max_eta = float(np.nanmax(finite)) if finite.size else None
    return {
        "incident_id": field.incident_id,
        "engine": field.engine,
        "spread_field_version": field.spread_field_version,
        "n_members": field.n_members,
        "west": field.west,
        "south": field.south,
        "east": field.east,
        "north": field.north,
        "width": sw,
        "height": sh,
        "stride": stride,
        "source_shape": [int(h), int(w)],
        "horizon_hours": 72.0,
        "max_eta_hours": max_eta,
        "arrival_hours": _grid(sub_a),
        "p_burn_72": _grid(sub_p),
    }
