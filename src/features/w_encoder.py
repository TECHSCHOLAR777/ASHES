"""Encode raw Mireye W fields into a typed, masked feature vector (SRS FR-17/FR-18, Step 6).

Rules (SRS 6.3 / `mireye_into_model.md`):
  - float: (x - mean) / std from a pre-computed scaler; missing -> impute 0, mask 0.
  - bool: 1.0 / 0.0; missing -> 0.5, mask 0.
  - ordered_categorical: fixed integer order from `config/field_sets.yaml`.
  - unordered_categorical: one-hot over the fixed category list, with an `other` bucket.
  - id / join_key fields: never a feature (join key only).
  - every field appends a paired `_conf` bit (1 = high/medium confidence, 0 = low/unverified).
  - most_recent_burn_year: encoded as years_since_burn; null -> mask 0, impute 50 (conservative).
  - role J (Response Agent fields) is never part of the model feature vector.

[NEW DECISION, see DECISIONS.md] No historical training corpus exists yet at build time, so
float scaling defaults to identity (mean=0, std=1) unless a persisted scaler is supplied by
`train.py`; passing a real scaler once one is trained is a drop-in replacement, the encoding
contract does not change.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

FIELD_SETS_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "field_sets.yaml"

CONFIDENCE_HIGH_VALUES = {"high", "medium"}


@dataclass
class WFeatures:
    vector: np.ndarray
    mask: np.ndarray
    citations: list[dict[str, Any]] = field(default_factory=list)
    vintages: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)


@functools.lru_cache(maxsize=1)
def load_field_catalog(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else FIELD_SETS_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ordered_model_fields(catalog: dict[str, Any] | None = None) -> list[tuple[str, str, dict[str, Any]]]:
    """Deterministic (role, field_name, meta) list for every field fed into h_fire.

    Excludes role J (Response Agent only), any field typed `id`, and any field flagged
    `join_key: true` -- those are join keys, never model features (SRS FR-17).
    """
    catalog = catalog or load_field_catalog()
    out: list[tuple[str, str, dict[str, Any]]] = []
    for role in sorted(catalog["roles"].keys()):
        role_cfg = catalog["roles"][role]
        if role_cfg.get("model_feature") is False:
            continue
        for field_name in sorted(role_cfg["fields"].keys()):
            meta = role_cfg["fields"][field_name]
            if meta.get("type") == "id" or meta.get("join_key"):
                continue
            out.append((role, field_name, meta))
    return out


def _confidence_bit(raw_w: dict[str, Any], field_name: str) -> float:
    conf = raw_w.get(f"{field_name}_confidence")
    if conf is None:
        return 0.0
    return 1.0 if str(conf).lower() in CONFIDENCE_HIGH_VALUES else 0.0


def _encode_float(value: Any, scaler: dict[str, tuple[float, float]] | None, field_name: str) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    mean, std = (scaler or {}).get(field_name, (0.0, 1.0))
    std = std if std else 1.0
    return (float(value) - mean) / std, 1.0


def _encode_bool(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.5, 0.0
    return (1.0 if bool(value) else 0.0), 1.0


def _encode_ordered_categorical(value: Any, order: list[Any]) -> tuple[float, float]:
    if value is None or value not in order:
        return 0.0, 0.0
    return float(order.index(value)), 1.0


def _category_slots(categories: list[str]) -> list[str]:
    """The one-hot slot names, adding an `other` bucket only if the catalog lacks one."""
    return categories if "other" in categories else [*categories, "other"]


def _encode_unordered_categorical(value: Any, categories: list[str]) -> tuple[np.ndarray, float]:
    slots = _category_slots(categories)
    vec = np.zeros(len(slots), dtype=np.float64)
    if value is None:
        return vec, 0.0
    if value in slots:
        vec[slots.index(value)] = 1.0
    else:
        vec[slots.index("other")] = 1.0
    return vec, 1.0


def _encode_burn_year(value: Any, now: datetime | None = None) -> tuple[float, float]:
    now = now or datetime.now(timezone.utc)
    if value is None:
        return 50.0, 0.0
    return float(now.year - int(value)), 1.0


def encode_w(
    raw_w: dict[str, Any],
    field_meta: dict[str, Any] | None = None,
    scaler: dict[str, tuple[float, float]] | None = None,
    now: datetime | None = None,
) -> WFeatures:
    """Encodes a raw Mireye W response into a typed, masked feature vector.

    `raw_w` is the flat dict returned by MireyeClient.fetch/fetch_batch, keyed by field
    name, with optional `<field>_confidence`, `<field>_source_url`, `<field>_vintage` siblings.
    """
    catalog = field_meta or load_field_catalog()
    fields = ordered_model_fields(catalog)

    values: list[float] = []
    masks: list[float] = []
    names: list[str] = []
    citations: list[dict[str, Any]] = []
    vintages: dict[str, Any] = {}

    for role, name, meta in fields:
        raw_value = raw_w.get(name)
        ftype = meta["type"]

        if name == "most_recent_burn_year":
            enc, present = _encode_burn_year(raw_value, now)
            values.append(enc)
            masks.append(present)
            names.append("years_since_burn")
        elif ftype == "float" or ftype == "int":
            enc, present = _encode_float(raw_value, scaler, name)
            values.append(enc)
            masks.append(present)
            names.append(name)
        elif ftype == "bool":
            enc, present = _encode_bool(raw_value)
            values.append(enc)
            masks.append(present)
            names.append(name)
        elif ftype == "ordered_categorical":
            enc, present = _encode_ordered_categorical(raw_value, meta["order"])
            values.append(enc)
            masks.append(present)
            names.append(name)
        elif ftype == "unordered_categorical":
            enc_vec, present = _encode_unordered_categorical(raw_value, meta["categories"])
            slots = _category_slots(meta["categories"])
            for i, v in enumerate(enc_vec):
                values.append(float(v))
                masks.append(present)
                names.append(f"{name}__{slots[i]}")
        else:
            continue

        conf_bit = _confidence_bit(raw_w, name)
        values.append(conf_bit)
        masks.append(1.0)
        names.append(f"{name}_conf")

        source_url = raw_w.get(f"{name}_source_url")
        vintage = raw_w.get(f"{name}_vintage")
        if raw_value is not None:
            if source_url:
                citations.append(
                    {
                        "source": "mireye",
                        "url": source_url,
                        "fetched_at": raw_w.get("fetched_at"),
                        "field": name,
                    }
                )
            if vintage is not None:
                vintages[name] = vintage

    return WFeatures(
        vector=np.array(values, dtype=np.float64),
        mask=np.array(masks, dtype=np.float64),
        citations=citations,
        vintages=vintages,
        feature_names=names,
    )
