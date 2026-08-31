"""Deterministic, versioned policy table (SRS FR-37, Step 9). Never the LLM.

Thresholds are loaded from `config/policy.yaml` so they are tunable without a code change
(SRS FR-37 closing line). Rule order matters: rules are checked top to bottom and the
first match wins, mirroring the ordering in SRS 5.9's baseline policy sketch, with the
sigma-suppression rule always applied last as a final safety clamp (NFR-17).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "policy.yaml"


@dataclass
class PolicyInput:
    """W/E-derived signals the policy needs beyond y_hat/sigma, all pre-computed upstream.

    None means "unknown/masked", never a fabricated default - the policy branches treat
    unknown conservatively (never escalates past `monitor` on missing egress/density data).
    """

    dist_perim_m: float
    red_flag: bool
    spc_elevated: bool
    firms_count_5km: int
    perimeter_unofficial: bool
    land_use_class: Optional[str] = None  # e.g. "Developed" - USFS LCMS Land_Use taxonomy
    ndvi_current: Optional[float] = None
    housing_density_per_km2: Optional[float] = None
    road_access_limited: Optional[bool] = None
    containment_pct: Optional[float] = None
    previously_in_play: bool = False
    eta_hours: Optional[float] = None
    eta_sigma_hours: Optional[float] = None


@dataclass
class PolicyResult:
    action: str
    reasons: list[str] = field(default_factory=list)
    policy_version: str = "unknown"
    flags: list[str] = field(default_factory=list)


@functools.lru_cache(maxsize=1)
def load_policy_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else POLICY_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_policy(y_hat: Optional[float], sigma: float, pin: PolicyInput, config: dict[str, Any] | None = None) -> PolicyResult:
    """First-match table. The agentic path passes y_hat=None; clock is ETA + distance + Mireye guards.

    y_hat remains on the signature so watch/h_fire tests can still exercise those branches.
    """
    cfg = config or load_policy_config()
    policy_version = cfg["version"]
    dist_km = pin.dist_perim_m / 1000.0
    reasons: list[str] = []
    flags: list[str] = []

    has_wfigs = pin.dist_perim_m < 999_999.0
    has_firms = pin.firms_count_5km > 0 or pin.dist_perim_m < 20_000

    # Rule 6: fire contained/gone and site was previously in play -> inspect_after.
    # Checked early: containment supersedes an otherwise-live-looking y_hat/E signal.
    if pin.containment_pct is not None and pin.containment_pct >= 1.0 and pin.previously_in_play:
        reasons.append("incident contained (100%) and site was previously in play")
        return PolicyResult("inspect_after", reasons, policy_version, flags)

    # Rule 1: no FIRMS persistence, no WFIGS, no Red Flag -> no_action (or monitor if SPC elevated).
    if not has_firms and not has_wfigs and not pin.red_flag:
        if pin.spc_elevated:
            reasons.append("no active fire signal, but SPC outlook is elevated")
            return PolicyResult("monitor", reasons, policy_version, flags)
        reasons.append("no FIRMS activity, no WFIGS incident, no Red Flag")
        return PolicyResult("no_action", reasons, policy_version, flags)

    # Rule 2: FIRMS without WFIGS -> monitor + perimeter_unofficial; never prepare on a
    # single urban hotspot (developed LCMS) - guards against flares/industry false positives.
    if has_firms and not has_wfigs:
        flags.append("perimeter_unofficial")
        if pin.land_use_class == "Developed":
            reasons.append("FIRMS hotspot on developed land with no WFIGS confirmation (likely non-wildfire source)")
        else:
            reasons.append("FIRMS hotspot detected with no WFIGS confirmation yet")
        return PolicyResult("monitor", reasons, policy_version, flags)

    dist_cfg = cfg["distance_thresholds"]
    y_cfg = cfg["y_hat_thresholds"]
    eta_cfg = cfg.get("eta_thresholds") or {}
    eta_sig_cfg = cfg.get("eta_sigma_thresholds") or {}
    evacuate_h = float(eta_cfg.get("evacuate_hours", 6))
    prepare_lo = float(eta_cfg.get("prepare_min_hours", 24))
    prepare_hi = float(eta_cfg.get("prepare_max_hours", 72))
    eta_sig_cut = float(eta_sig_cfg.get("evacuate_suppress_hours", 12))

    # Rule 3: WFIGS distance > 15 km and low y_hat -> monitor.
    if has_wfigs and dist_km > dist_cfg["monitor_only_km"] and (y_hat or 0.0) < 0.3:
        if pin.eta_hours is None or pin.eta_hours > prepare_hi:
            reasons.append(f"perimeter is {dist_km:.1f} km away and model score is low")
            return PolicyResult("monitor", reasons, policy_version, flags)

    # Rule 4: dist 4-10 km + Red Flag + high fuel W (NDVI) + moderate y_hat -> prepare.
    high_ndvi = pin.ndvi_current is not None and pin.ndvi_current > cfg["ndvi_high_fuel_threshold"]
    if (
        dist_cfg["prepare_min_km"] <= dist_km <= dist_cfg["prepare_max_km"]
        and pin.red_flag
        and high_ndvi
        and (y_hat or 0.0) > y_cfg["prepare"]
    ):
        reasons.append(
            f"perimeter {dist_km:.1f} km away, Red Flag active, high fuel continuity (NDVI), y_hat={y_hat:.2f}"
        )
        return PolicyResult("prepare", reasons, policy_version, flags)

    # FR-38: ETA in the 24-72 h window is prepare, unless a closer/higher-score
    # protect/evacuate rule below already applies.
    eta_prepare = (
        pin.eta_hours is not None and prepare_lo <= pin.eta_hours <= prepare_hi
    )
    eta_imminent = pin.eta_hours is not None and pin.eta_hours < evacuate_h

    # Rule 5: dist small or high y_hat or imminent ETA -> protect_asset / evacuate_site.
    if dist_km < dist_cfg["protect_km"] or (y_hat or 0.0) > y_cfg["protect"] or eta_imminent:
        high_density = (
            pin.housing_density_per_km2 is not None
            and pin.housing_density_per_km2 > cfg["housing_density_evacuate_threshold_per_km2"]
        )
        limited_egress = bool(pin.road_access_limited)
        if eta_imminent and high_density and limited_egress:
            reasons.append(f"ETA {pin.eta_hours:.1f} h with high housing density and limited egress")
            action = "evacuate_site"
        elif high_density and limited_egress:
            reasons.append(
                f"perimeter {dist_km:.1f} km away or y_hat={y_hat}, high housing density and limited road egress"
            )
            action = "evacuate_site"
        elif eta_imminent:
            reasons.append(f"ETA {pin.eta_hours:.1f} h (arrival estimate with sigma); protecting the asset")
            action = "protect_asset"
        else:
            reasons.append(f"perimeter {dist_km:.1f} km away or y_hat={y_hat}, protecting the asset")
            action = "protect_asset"

        # Rule 7: high sigma (y_hat or ETA) suppresses an overconfident evacuate (NFR-17 / FR-38).
        sigma_cfg = cfg["sigma_thresholds"]
        high_eta_sigma = pin.eta_sigma_hours is not None and pin.eta_sigma_hours >= eta_sig_cut
        if action == "evacuate_site" and (sigma >= sigma_cfg["evacuate_suppress"] or high_eta_sigma):
            flags.append("no_ros_high_sigma")
            if high_eta_sigma:
                reasons.append(
                    f"eta_sigma_hours={pin.eta_sigma_hours:.1f} too high for an overconfident evacuate; downgraded to protect_asset"
                )
            else:
                reasons.append(f"sigma={sigma:.2f} too high for an overconfident evacuate; downgraded to protect_asset")
            action = "protect_asset"

        return PolicyResult(action, reasons, policy_version, flags)

    if eta_prepare:
        reasons.append(f"ETA {pin.eta_hours:.1f} h is inside the 24-72 h prepare window (arrival estimate with sigma)")
        return PolicyResult("prepare", reasons, policy_version, flags)

    # Fallback: fire signal present but none of the escalation thresholds met -> monitor.
    reasons.append("active fire signal present but below prepare/protect thresholds")
    return PolicyResult("monitor", reasons, policy_version, flags)
