"""Builds real (site, event, t0, y) training samples from MTBS final perimeters (SRS 6.2).

This is genuine historical data collection, not synthetic bootstrap: MTBS gives the gold
label (site inside the final perimeter), and each sample's W/E is reconstructed from real
Mireye/FIRMS-archive/HRRR-archive calls at the fire's own vintage where possible.

HONEST LIMITATION (read before trusting these labels for a real H1/H2 claim - see
DECISIONS.md and README "Known limitations"): MTBS only publishes the fire's ignition
DATE and FINAL perimeter, not a perimeter time series. `dist_perim_m` here is computed to
the fire's ignition centroid (`dist_source=ignition_centroid_proxy`), not to "the perimeter
as it existed at t0" (SRS 6.3's actual E-feature definition). NIFC Interagency Fire
Perimeter History is a *final* mapped perimeter (`FEATURE_CA` typically "Wildfire Final
Fire Perimeter", live-verified 2026-08-27); using it as t0 distance would leak the
outcome (SRS 6.1). V2's `--with-spread` path attaches a delegated arrival-field sample
instead of trying to fake a t0 perimeter. `acres`/`containment_pct` are left null
(masked), not backfilled from the final MTBS acreage, for the same leakage reason.
Vintage-dynamic W fields (`ndvi_current`, `ndvi_change_5y`, `drought_category`) are
stripped before encoding per SRS 6.2 ("never use 2026 NDVI on a 2018 fire") since Mireye
only serves current values.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from pyproj import Geod

from src.clients.firms import FIRMSClient
from src.clients.hrrr import HRRRClient
from src.clients.mireye import MireyeClient
from src.clients.mtbs import MTBSFire, point_in_multipolygon
from src.features.e_packer import EFeatures, e_features_to_vector, pack_e
from src.features.ros_ellipse import compute_ros_ellipse_feature
from src.features.w_encoder import encode_w, load_field_catalog, ordered_model_fields

logger = logging.getLogger("fire_copilot.training_data")

_GEOD = Geod(ellps="WGS84")

# t0 offset after ignition: MTBS gives only a date, so t0 is pinned to noon UTC on the
# ignition day - an arbitrary but documented and consistent choice, not a real discovery
# timestamp (SRS 6.2's t0 is normally the incident discovery time, which MTBS doesn't carry).
T0_HOUR_UTC = 12

HARD_NEGATIVE_MIN_KM = 2.0
HARD_NEGATIVE_MAX_KM = 20.0


def ignition_datetime(fire: MTBSFire) -> datetime | None:
    if fire.ignition_date is None:
        return None
    return datetime.combine(fire.ignition_date, time(hour=T0_HOUR_UTC), tzinfo=timezone.utc)


def _bbox_of_rings(geometry_rings: list[list[list[tuple[float, float]]]]) -> tuple[float, float, float, float]:
    lngs = [pt[0] for polygon in geometry_rings for ring in polygon for pt in ring]
    lats = [pt[1] for polygon in geometry_rings for ring in polygon for pt in ring]
    return min(lngs), min(lats), max(lngs), max(lats)


def sample_positive_points(fire: MTBSFire, n: int, rng: random.Random, max_attempts: int = 2000) -> list[tuple[float, float]]:
    """Rejection-samples n points inside the fire's final perimeter (the V1 gold label)."""
    if not fire.geometry_rings:
        return []
    min_lng, min_lat, max_lng, max_lat = _bbox_of_rings(fire.geometry_rings)
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        lat = rng.uniform(min_lat, max_lat)
        lng = rng.uniform(min_lng, max_lng)
        if point_in_multipolygon(lat, lng, fire.geometry_rings):
            points.append((lat, lng))
    return points


def sample_hard_negative_points(fire: MTBSFire, n: int, rng: random.Random, max_attempts: int = 2000) -> list[tuple[float, float]]:
    """Points 2-20 km outside the perimeter, radiating from the fire's own centroid."""
    if fire.centroid_lat is None or fire.centroid_lng is None:
        return []
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        bearing = rng.uniform(0, 360)
        dist_km = rng.uniform(HARD_NEGATIVE_MIN_KM, HARD_NEGATIVE_MAX_KM)
        lng, lat, _ = _GEOD.fwd(fire.centroid_lng, fire.centroid_lat, bearing, dist_km * 1000.0)
        if not point_in_multipolygon(lat, lng, fire.geometry_rings):
            points.append((lat, lng))
    return points


def sample_easy_negative_points(
    other_fire_centroids: list[tuple[float, float]], n: int, rng: random.Random, min_km: float = 100.0
) -> list[tuple[float, float]]:
    """Points far from any known fire (a different ecoregion proxy, per SRS 6.2)."""
    if not other_fire_centroids:
        return []
    points: list[tuple[float, float]] = []
    for _ in range(n):
        base_lat, base_lng = rng.choice(other_fire_centroids)
        bearing = rng.uniform(0, 360)
        dist_km = rng.uniform(min_km, min_km * 2)
        lng, lat, _ = _GEOD.fwd(base_lng, base_lat, bearing, dist_km * 1000.0)
        points.append((lat, lng))
    return points


def _strip_dynamic_vintage_fields(raw_w: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Removes fields Mireye can only serve at today's vintage, never a historical one
    (SRS 6.2). Encoded downstream as masked/imputed, exactly like a live missing field."""
    dynamic_fields = [
        name
        for role in catalog["roles"].values()
        for name, meta in role["fields"].items()
        if meta.get("vintage_role") == "dynamic"
    ]
    stripped = dict(raw_w)
    for name in dynamic_fields:
        for suffix in ("", "_confidence", "_source_url", "_vintage"):
            stripped.pop(f"{name}{suffix}", None)
    return stripped


def dist_to_ignition_centroid_m(site_lat: float, site_lng: float, fire: MTBSFire) -> float:
    if fire.centroid_lat is None or fire.centroid_lng is None:
        return 999_999.0
    _, _, dist_m = _GEOD.inv(site_lng, site_lat, fire.centroid_lng, fire.centroid_lat)
    return dist_m


@dataclass
class SampleBuildResult:
    site_id: str
    event_id: str
    t0: str
    w_vector: list[float]
    w_mask: list[float]
    e_vector: list[float]
    y: int
    dist_source: str = "ignition_centroid_proxy"
    spread_vector: list[float] | None = None


def build_sample(
    mireye: MireyeClient,
    firms: FIRMSClient,
    hrrr: HRRRClient,
    fire: MTBSFire,
    site_lat: float,
    site_lng: float,
    label: int,
    sample_id: str,
) -> SampleBuildResult | None:
    """Builds one training sample. Returns None if a required real fetch fails outright
    (never fabricates the missing data to force a sample through)."""
    t0 = ignition_datetime(fire)
    if t0 is None:
        return None

    catalog = load_field_catalog()
    all_fields = [name for _, name, _ in ordered_model_fields(catalog)]
    # ordered_model_fields returns encoder-stage names (e.g. "years_since_burn" is derived
    # from most_recent_burn_year); fetch by the real Mireye field names instead.
    fetch_fields = sorted(
        {
            field_name
            for role in catalog["roles"].values()
            if role.get("model_feature") is not False
            for field_name, meta in role["fields"].items()
            if meta.get("type") != "id" and not meta.get("join_key")
        }
    )

    try:
        raw_w = mireye.fetch(site_lat, site_lng, fetch_fields, site_id=sample_id)
    except Exception as exc:
        logger.warning("Mireye fetch failed for sample %s: %s; skipping sample", sample_id, exc)
        return None
    raw_w = _strip_dynamic_vintage_fields(raw_w, catalog)
    w_features = encode_w(raw_w, catalog, now=t0)

    firms_result = firms.get_historical_hotspots(site_lat, site_lng, on_date=t0.date(), site_id=sample_id)

    try:
        hrrr_weather = hrrr.get_weather(site_lat, site_lng, site_id=sample_id, at=t0)
    except Exception as exc:
        logger.warning("HRRR archive fetch failed for sample %s: %s; wind features masked", sample_id, exc)
        hrrr_weather = None

    dist_m = dist_to_ignition_centroid_m(site_lat, site_lng, fire)
    dist_source = "ignition_centroid_proxy"
    ros_dist = compute_ros_ellipse_feature(
        site_lat=site_lat,
        site_lng=site_lng,
        fire_lat=fire.centroid_lat,
        fire_lng=fire.centroid_lng,
        wind_u=hrrr_weather.wind_u_10m if hrrr_weather else None,
        wind_v=hrrr_weather.wind_v_10m if hrrr_weather else None,
        acres=None,
        dist_perim_m=dist_m,
    )

    e_features: EFeatures = pack_e(
        firms=firms_result,
        wfigs_incident=None,  # no historical incident progression data available (see module docstring)
        wfigs_perim_dist_m=dist_m,
        cap_alerts=[],  # no historical CAP alert archive used (see module docstring)
        spc=None,
        hrrr=hrrr_weather,
        ros_ellipse_dist_m=ros_dist,
    )
    e_vector = e_features_to_vector(e_features)

    return SampleBuildResult(
        site_id=sample_id,
        event_id=fire.event_id,
        t0=t0.isoformat(),
        w_vector=w_features.vector.tolist(),
        w_mask=w_features.mask.tolist(),
        e_vector=e_vector.tolist(),
        y=label,
        dist_source=dist_source,
        spread_vector=None,
    )


def attach_spread_vector(result: SampleBuildResult, spread) -> SampleBuildResult:
    """Copy a delegated-field sample onto a training row. Does not invent values."""
    if spread is None:
        return result
    result.spread_vector = [
        float(spread.eta_hours) if spread.eta_hours is not None else 72.0,
        float(spread.eta_sigma_hours) if spread.eta_sigma_hours is not None else 24.0,
        float(spread.p_burn_24) if spread.p_burn_24 is not None else 0.0,
        float(spread.p_burn_48) if spread.p_burn_48 is not None else 0.0,
        float(spread.p_burn_72) if spread.p_burn_72 is not None else 0.0,
    ]
    return result
