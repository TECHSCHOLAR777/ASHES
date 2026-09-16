"""Vintage-safe W columns for the Job C calibrator (not the fetch).

Mireye is still queried for roles A–I. The head that claims
``P(y | eta, σ, p_T, W_allowed)`` must not see post-fire vegetation, a
distance-to-scar leak, or vintage-dynamic fields that are not a 2023 time
machine. Role J stays Response-Agent-only (already dropped by ``encode_w``).
"""
from __future__ import annotations

from typing import Any

from src.features.w_encoder import load_field_catalog, model_feature_layout, ordered_model_fields

# Distance-to-scar / post-fire fuel. Also listed in config/field_sets.yaml job_c.
DEFAULT_EXCLUDE = frozenset(
    {
        "ndvi_current",
        "ndvi_change_5y",
        "drought_category",
        "nearest_fire_perimeter_distance_m",
        "most_recent_burn_year",
        "lcms_class",
        "tree_canopy_pct",
    }
)


def excluded_fields(catalog: dict[str, Any] | None = None) -> set[str]:
    catalog = catalog or load_field_catalog()
    named = set((catalog.get("job_c") or {}).get("head_exclude_fields") or [])
    out = set(DEFAULT_EXCLUDE) | named
    for _role, name, meta in ordered_model_fields(catalog):
        if meta.get("vintage_role") == "dynamic":
            out.add(name)
    return out


def allowed_fields(catalog: dict[str, Any] | None = None) -> list[str]:
    catalog = catalog or load_field_catalog()
    skip = excluded_fields(catalog)
    return [name for _role, name, _meta in ordered_model_fields(catalog) if name not in skip]


def allowed_column_indices(
    catalog: dict[str, Any] | None = None,
    include_fields: list[str] | None = None,
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    """Indices into the 200-D encoded W vector, plus the column names and field spans.

    ``include_fields`` further subsets W_allowed. Excluded leaks/t0-fuel cannot
    be re-introduced here — collection still stores the full fetch.
    """
    catalog = catalog or load_field_catalog()
    skip = excluded_fields(catalog)
    allowed = set(allowed_fields(catalog))
    if include_fields is not None:
        wanted = []
        for name in include_fields:
            if name in skip or name not in allowed:
                raise ValueError(f"field {name!r} is not in W_allowed")
            wanted.append(name)
        keep = set(wanted)
    else:
        keep = None
    layout = model_feature_layout(catalog)
    indices: list[int] = []
    names: list[str] = []
    kept_layout: list[dict[str, Any]] = []
    for row in layout:
        if row["field"] in skip:
            continue
        if keep is not None and row["field"] not in keep:
            continue
        kept_layout.append(row)
        for offset, col in enumerate(row["columns"]):
            indices.append(row["start"] + offset)
            names.append(col)
    if keep is not None and {row["field"] for row in kept_layout} != keep:
        missing = keep - {row["field"] for row in kept_layout}
        raise ValueError(f"include_fields not in encoded layout: {sorted(missing)}")
    return indices, names, kept_layout


def slice_w(w_vector, w_mask, indices: list[int] | None = None):
    """Masked W_allowed row. Missing bits stay 0 via the mask."""
    import numpy as np

    if indices is None:
        indices, _, _ = allowed_column_indices()
    w = np.asarray(w_vector, dtype=np.float64)
    m = np.asarray(w_mask, dtype=np.float64)
    idx = np.asarray(indices, dtype=int)
    return w[idx] * m[idx]
