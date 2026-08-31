"""Pack live E feeds (FIRMS, WFIGS, NWS, SPC, HRRR, ROS ellipse) into the E feature block.

SRS FR-13 / 6.3. Every field here is either a raw copy from a feed (never invented) or a
deterministic function of those copies (geodesic distance, ROS ellipse). No LLM involvement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.clients.firms import FIRMSResult
from src.clients.hrrr import HRRRWeather
from src.clients.nws import CAPAlert, SPCOutlook
from src.clients.wfigs import WFIGSIncident, WFIGSPerimeter

FAR_SENTINEL_M = 999_999.0


@dataclass
class EFeatures:
    rflag: bool
    spc_day1_elevated: bool
    firms_count_5km: int
    firms_count_10km: int
    firms_count_20km: int
    frp_sum_5km: float
    dist_perim_m: float
    acres: float
    containment_pct: float
    hours_since_discovery: float | None
    perimeter_unofficial: bool
    wind_speed_ms: float | None
    wind_dir_deg: float | None
    temp_c: float | None
    rh_pct: float | None
    wind_ros_ellipse_dist_m: float
    firms_unavailable: bool = False
    stale_e: bool = False


def _hours_since(discovery_iso: str | None, now_iso: str | None = None) -> float | None:
    if not discovery_iso:
        return None
    from datetime import datetime, timezone

    try:
        discovery = datetime.fromisoformat(discovery_iso.replace("Z", "+00:00"))
        now = datetime.fromisoformat(now_iso.replace("Z", "+00:00")) if now_iso else datetime.now(timezone.utc)
        return (now - discovery).total_seconds() / 3600.0
    except (ValueError, AttributeError):
        return None


def pack_e(
    firms: FIRMSResult | None,
    wfigs_incident: WFIGSIncident | None,
    wfigs_perim_dist_m: float | None,
    cap_alerts: list[CAPAlert],
    spc: SPCOutlook | None,
    hrrr: HRRRWeather | None,
    ros_ellipse_dist_m: float,
) -> EFeatures:
    rflag = any(a.is_red_flag for a in cap_alerts)
    spc_elevated = bool(spc and spc.elevated)

    firms_unavailable = firms.unavailable if firms else True
    count_5km = firms.count_5km if firms else 0
    count_10km = firms.count_10km if firms else 0
    count_20km = firms.count_20km if firms else 0
    frp_sum_5km = firms.frp_sum_5km if firms else 0.0

    dist_perim_m = wfigs_perim_dist_m if wfigs_perim_dist_m is not None else FAR_SENTINEL_M
    acres = wfigs_incident.acres if (wfigs_incident and wfigs_incident.acres is not None) else 0.0
    containment_pct = (
        wfigs_incident.containment_pct
        if (wfigs_incident and wfigs_incident.containment_pct is not None)
        else 0.0
    )
    hours_since_discovery = _hours_since(wfigs_incident.discovery_datetime) if wfigs_incident else None
    # True whenever any fire signal is present that is not an MTBS-final perimeter: a WFIGS
    # incident/perimeter (always operational, SRS 2.2.2) or FIRMS-only activity (SRS FR-8).
    has_firms_activity = count_5km > 0 or count_10km > 0 or count_20km > 0
    perimeter_unofficial = wfigs_incident is not None or dist_perim_m < FAR_SENTINEL_M or has_firms_activity

    return EFeatures(
        rflag=rflag,
        spc_day1_elevated=spc_elevated,
        firms_count_5km=count_5km,
        firms_count_10km=count_10km,
        firms_count_20km=count_20km,
        frp_sum_5km=frp_sum_5km,
        dist_perim_m=dist_perim_m,
        acres=acres,
        containment_pct=containment_pct,
        hours_since_discovery=hours_since_discovery,
        perimeter_unofficial=perimeter_unofficial,
        wind_speed_ms=hrrr.wind_speed_ms if hrrr else None,
        wind_dir_deg=None,
        temp_c=hrrr.temp_c if hrrr else None,
        rh_pct=hrrr.rh_pct if hrrr else None,
        wind_ros_ellipse_dist_m=ros_ellipse_dist_m,
        firms_unavailable=firms_unavailable,
        stale_e=bool(hrrr.stale) if hrrr else False,
    )


# Fixed, documented order: this is the E half of the [w_vector*w_mask ; e_vector]
# concatenation h_fire trains and infers on (SRS 6.3, Step 8 train.py). Changing this order
# requires retraining every saved model artefact.
E_VECTOR_FIELD_ORDER = [
    "rflag",
    "spc_day1_elevated",
    "firms_count_5km",
    "firms_count_10km",
    "firms_count_20km",
    "frp_sum_5km",
    "dist_perim_m",
    "acres",
    "containment_pct",
    "hours_since_discovery",
    "perimeter_unofficial",
    "wind_speed_ms",
    "temp_c",
    "rh_pct",
    "wind_ros_ellipse_dist_m",
]


def e_features_to_vector(e: EFeatures) -> np.ndarray:
    """Flattens EFeatures into the fixed-order numeric vector fed to h_fire.

    Missing optionals (wind/temp/rh/hours_since_discovery) impute to 0.0 - the model
    also receives `firms_unavailable`/`stale_e` as ActionCard flags, so these zeros are
    never presented to a human as measured values, only to the model as a neutral default.
    """
    return np.array(
        [
            1.0 if e.rflag else 0.0,
            1.0 if e.spc_day1_elevated else 0.0,
            float(e.firms_count_5km),
            float(e.firms_count_10km),
            float(e.firms_count_20km),
            float(e.frp_sum_5km),
            float(e.dist_perim_m),
            float(e.acres),
            float(e.containment_pct),
            float(e.hours_since_discovery) if e.hours_since_discovery is not None else 0.0,
            1.0 if e.perimeter_unofficial else 0.0,
            float(e.wind_speed_ms) if e.wind_speed_ms is not None else 0.0,
            float(e.temp_c) if e.temp_c is not None else 0.0,
            float(e.rh_pct) if e.rh_pct is not None else 0.0,
            float(e.wind_ros_ellipse_dist_m),
        ],
        dtype=np.float64,
    )
