"""Main prediction agent: watch and ask modes (SRS 2.2.4, FR-29/FR-33, Step 10).

Executes the fixed tool order from FR-30: geocode (if address) -> nws_alerts -> (firms +
wfigs incidents + wfigs perimeters + hrrr in parallel) -> pack E -> quote/fetch W ->
model_infer -> policy -> ActionCard -> LLM brief -> deliver. Every external call is wrapped
so a failure degrades the card with a flag instead of raising (FR-33); the LLM never
computes a number, only writes copy-only prose (FR-36).

[NEW DECISION, see DECISIONS.md] "In parallel" for the four E fetches (FR-30) is
implemented with a `ThreadPoolExecutor` rather than `asyncio.gather`, because the client
classes (`httpx.Client`, not `AsyncClient`) are synchronous; a thread pool gives the same
real concurrency without doubling every client into sync+async variants for a V1 build.
"""
from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, get_args

import yaml

from src.cache.w_cache import WCache
from src.clients.firms import FIRMSClient, FIRMSResult
from src.clients.hrrr import HRRRClient, HRRRWeather
from src.clients.landfire import LANDFIREClient, LandfireRequestFailed
from src.clients.mireye import MireyeClient
from src.clients.nws import NWSClient
from src.clients.osm import OSMClient
from src.clients.usgs import USGSClient
from src.clients.wfigs import WFIGSClient, WFIGSIncident, distance_to_perimeter_m, geodesic_distance_m
from src.delivery.email_delivery import deliver_cards_by_email
from src.delivery.slack_delivery import deliver_action_card
from src.features.e_packer import pack_e
from src.features.ros_ellipse import compute_ros_ellipse_feature
from src.features.w_encoder import load_field_catalog, ordered_model_fields, encode_w
from src.geometry.aoi import Geometry, aoi_bbox_for_fetch, build_incident_aoi, point_geometry
from src.logging_ import tool_logger
from src.model.h_fire import model_infer
from src.policy.engine import PolicyInput, apply_policy, load_policy_config
from src.schemas.action_card import (
    ActionCard,
    Citation,
    EProductTime,
    FlagEnum,
    IncidentInfo,
    PBurnByT,
    SiteRef,
    WeatherInfo,
    WSource,
)
from src.spread.client import SpreadSiteSample, spread_run
from src.state.site_state import SiteStateStore
from src.validator.brief_validator import safe_brief

logger = logging.getLogger("fire_copilot.main_agent")

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")

BRIEF_SYSTEM_PROMPT = (
    "You are a fire brief writer. You receive a fully populated ActionCard JSON. Write a "
    "2-3 paragraph plain-English brief that copies the values from the card verbatim. Do not "
    "invent any new numbers, distances, acres, containment percentages, or scores. Do not "
    "alter the action or severity. If the action is evacuate_site, lead with that. Return "
    "only the prose."
)

RECOMMENDED_ACTIONS: dict[str, list[str]] = {
    "no_action": ["Continue routine monitoring cadence."],
    "monitor": ["Increase poll frequency awareness; no action needed from site staff yet."],
    "prepare": [
        "Stage evacuation-ready documents and critical equipment.",
        "Confirm site contact list and communication channels are current.",
    ],
    "protect_asset": [
        "Deploy available fire-suppression resources at the site.",
        "Clear defensible space around structures if time permits.",
        "Confirm the Response Dossier's water sources and access routes with on-site staff.",
    ],
    "evacuate_site": [
        "Evacuate all personnel immediately via the routes in the Response Dossier.",
        "Notify local emergency services of site occupancy status.",
    ],
    "inspect_after": ["Schedule a post-incident structural and environmental inspection."],
}


@dataclass
class Site:
    site_id: str
    name: str
    lat: float
    lng: float
    address: Optional[str] = None
    slack_channel: str = "#fire-alerts"
    email: Optional[str] = None


def load_sites(path: str | None = None) -> list[Site]:
    p = path or os.path.join(CONFIG_DIR, "sites.yaml")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Site(**s) for s in data["sites"]]


class MainAgentDeps:
    """Bundles the client/cache singletons the agent needs, built once and reused."""

    def __init__(self) -> None:
        keys = [
            os.environ.get("MIREYE_KEY_1"),
            os.environ.get("MIREYE_KEY_2"),
            os.environ.get("MIREYE_KEY_3"),
        ]
        keys = [k for k in keys if k]
        if not keys:
            raise RuntimeError("No MIREYE_KEY_* found in environment; cannot enrich W")
        self.mireye = MireyeClient(keys)
        self.nws = NWSClient()
        self.firms = FIRMSClient()
        self.wfigs = WFIGSClient()
        self.hrrr = HRRRClient()
        self.usgs = USGSClient()
        self.landfire = LANDFIREClient()
        self.osm = OSMClient()
        self.w_cache = WCache()
        self.site_state = SiteStateStore()
        self.field_catalog = load_field_catalog()


def _fetch_w_for_site(deps: MainAgentDeps, site: Site) -> tuple[dict[str, Any], list[str]]:
    """Reads role A-I fields from cache where fresh, fetches the rest from Mireye in one call."""
    roles = [r for r in sorted(deps.field_catalog["roles"].keys()) if r != "J"]
    raw_w: dict[str, Any] = {}
    missing_fields: list[str] = []
    missing_roles: list[str] = []

    for role in roles:
        cached = deps.w_cache.get(site.site_id, role)
        if cached is not None:
            raw_w.update(cached.fields)
        else:
            missing_roles.append(role)
            role_fields = list(deps.field_catalog["roles"][role]["fields"].keys())
            missing_fields.extend(role_fields)

    flags: list[str] = []
    if missing_fields:
        try:
            fetched = deps.mireye.fetch(site.lat, site.lng, missing_fields, site_id=site.site_id)
            raw_w.update(fetched)
            now = datetime.now(timezone.utc)
            for role in missing_roles:
                role_fields = list(deps.field_catalog["roles"][role]["fields"].keys())
                role_data = {k: fetched.get(k) for k in role_fields if k in fetched}
                vintages = {k: fetched.get(f"{k}_vintage") for k in role_fields if fetched.get(f"{k}_vintage")}
                deps.w_cache.put(site.site_id, role, role_data, vintages, site.lat, site.lng, now=now)
        except Exception as exc:
            logger.warning("Mireye W fetch failed for %s: %s; continuing with cached/degraded W", site.site_id, exc)
            flags.append("degraded")

    if raw_w.get("fire_hazard_severity_zone_class") is None:
        flags.append("fhsz_missing")

    return raw_w, flags


def _nearest_incident_and_perimeter_distance(
    site: Site, incidents: list[WFIGSIncident], perimeters: list
) -> tuple[Optional[WFIGSIncident], Optional[float]]:
    nearest_incident = None
    if incidents:
        nearest_incident = min(incidents, key=lambda i: geodesic_distance_m(site.lat, site.lng, i.lat, i.lng))

    dist_perim_m = None
    if perimeters:
        dist_perim_m = min(distance_to_perimeter_m(site.lat, site.lng, p) for p in perimeters)

    return nearest_incident, dist_perim_m


def _write_llm_brief(card: ActionCard) -> str:
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        return _fallback_brief(card)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
                {"role": "user", "content": card.model_dump_json()},
            ],
            max_tokens=400,
        )
        return resp.choices[0].message.content or _fallback_brief(card)
    except Exception as exc:
        logger.warning("OpenAI brief generation failed: %s; using fallback brief", exc)
        return _fallback_brief(card)


def _fallback_brief(card: ActionCard) -> str:
    return (
        f"Action: {card.action}. Site: {card.site.name}. "
        f"Model score y_hat={card.y_hat} (sigma={card.sigma}), baseline={card.baseline_y}. "
        f"Distance to perimeter: {card.incident.dist_perimeter_m} m. "
        f"Reasons: {'; '.join(card.reasons)}."
    )


def build_action_card(deps: MainAgentDeps, site: Site) -> ActionCard:
    now = datetime.now(timezone.utc)
    flags: list[str] = []

    if site.address and (site.lat is None or site.lng is None):
        geo = deps.mireye.geocode(site.address)
        site.lat, site.lng = geo.lat, geo.lng
        geocode_confidence = geo.confidence
        range_interpolation = geo.range_interpolation
    else:
        geocode_confidence = "n/a"
        range_interpolation = False

    cap_alerts = deps.nws.get_cap_alerts(site.lat, site.lng, site_id=site.site_id)
    spc = deps.nws.get_spc_outlook(site_id=site.site_id)

    firms_result: Optional[FIRMSResult] = None
    incidents: list[WFIGSIncident] = []
    perimeters: list = []
    hrrr_weather: Optional[HRRRWeather] = None

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_firms = pool.submit(deps.firms.get_hotspots, site.lat, site.lng, 1, site.site_id)
        f_incidents = pool.submit(deps.wfigs.get_incidents, site.lat, site.lng, 100, site.site_id)
        f_perimeters = pool.submit(deps.wfigs.get_perimeters, site.lat, site.lng, 100, site.site_id)
        f_hrrr = pool.submit(deps.hrrr.get_weather, site.lat, site.lng, site.site_id)

        try:
            firms_result = f_firms.result()
        except Exception as exc:
            logger.warning("FIRMS fetch failed for %s: %s", site.site_id, exc)
            flags.append("firms_unavailable")
        try:
            incidents = f_incidents.result()
        except Exception as exc:
            logger.warning("WFIGS incidents fetch failed for %s: %s", site.site_id, exc)
        try:
            perimeters = f_perimeters.result()
        except Exception as exc:
            logger.warning("WFIGS perimeters fetch failed for %s: %s", site.site_id, exc)
        try:
            hrrr_weather = f_hrrr.result()
        except Exception as exc:
            logger.warning("HRRR fetch failed for %s: %s", site.site_id, exc)
            flags.append("stale_E")

    nearest_incident, dist_perim_m = _nearest_incident_and_perimeter_distance(site, incidents, perimeters)

    ros_dist = compute_ros_ellipse_feature(
        site_lat=site.lat,
        site_lng=site.lng,
        fire_lat=nearest_incident.lat if nearest_incident else None,
        fire_lng=nearest_incident.lng if nearest_incident else None,
        wind_u=hrrr_weather.wind_u_10m if hrrr_weather else None,
        wind_v=hrrr_weather.wind_v_10m if hrrr_weather else None,
        acres=nearest_incident.acres if nearest_incident else None,
        dist_perim_m=dist_perim_m if dist_perim_m is not None else 999_999.0,
    )

    e_features = pack_e(
        firms=firms_result,
        wfigs_incident=nearest_incident,
        wfigs_perim_dist_m=dist_perim_m,
        cap_alerts=cap_alerts,
        spc=spc,
        hrrr=hrrr_weather,
        ros_ellipse_dist_m=ros_dist,
    )
    if e_features.firms_unavailable:
        flags.append("firms_unavailable")
    if e_features.perimeter_unofficial:
        flags.append("perimeter_unofficial")
    if e_features.stale_e:
        flags.append("stale_E")

    raw_w, w_flags = _fetch_w_for_site(deps, site)
    flags.extend(w_flags)

    w_features = encode_w(raw_w, deps.field_catalog, now=now)

    spread_sample, geom = _try_spread(deps, site, nearest_incident, perimeters, hrrr_weather, flags)

    model_output = model_infer(site.site_id, w_features, e_features, w_features.vintages, spread=spread_sample)
    tool_logger.log_model_call(
        site.site_id,
        w_features.vector.shape,
        (15,),
        model_output.y_hat,
        model_output.sigma,
        model_output.baseline_y,
        model_output.model_version,
        eta_hours=model_output.eta_hours,
        spread_field_version=model_output.spread_field_version,
    )

    prior_state = deps.site_state.get_state(site.site_id)
    previously_in_play = bool(prior_state and prior_state.last_action not in (None, "no_action", "monitor"))

    road_access_limited = raw_w.get("nearest_road_class") in ("unclassified", "service", "track", "unknown", None)
    pin = PolicyInput(
        dist_perim_m=e_features.dist_perim_m,
        red_flag=e_features.rflag,
        spc_elevated=e_features.spc_day1_elevated,
        firms_count_5km=e_features.firms_count_5km,
        perimeter_unofficial=e_features.perimeter_unofficial,
        land_use_class=raw_w.get("land_use_class"),
        ndvi_current=raw_w.get("ndvi_current"),
        housing_density_per_km2=raw_w.get("housing_units_density_per_km2"),
        road_access_limited=road_access_limited,
        containment_pct=e_features.containment_pct,
        previously_in_play=previously_in_play,
        eta_hours=model_output.eta_hours,
        eta_sigma_hours=model_output.eta_sigma_hours,
    )
    policy_result = apply_policy(model_output.y_hat, model_output.sigma, pin)
    tool_logger.log_policy_call(
        site.site_id, model_output.y_hat, model_output.sigma, policy_result.action, policy_result.reasons, policy_result.policy_version
    )
    flags.extend(policy_result.flags)

    citations = [Citation(**c) for c in w_features.citations]
    w_sources = [
        WSource(field=field_name, source_url=raw_w.get(f"{field_name}_source_url", "n/a"), vintage=str(vintage), confidence=raw_w.get(f"{field_name}_confidence"))
        for field_name, vintage in w_features.vintages.items()
    ]
    e_product_times = [
        EProductTime(feed="nws_cap", product_time=now.isoformat()),
        EProductTime(feed="firms", product_time=now.isoformat()),
        EProductTime(feed="wfigs", product_time=now.isoformat()),
        EProductTime(feed="hrrr", product_time=hrrr_weather.hrrr_valid_time if hrrr_weather else now.isoformat()),
    ]

    card = ActionCard(
        card_id=str(uuid.uuid4()),
        generated_at=now.isoformat(),
        site=SiteRef(
            site_id=site.site_id,
            name=site.name,
            lat=site.lat,
            lng=site.lng,
            mode="aoi" if geom.mode == "aoi" and geom.contains_point(site.lat, site.lng) else "point",
            geocode_confidence=geocode_confidence,
            range_interpolation=range_interpolation,
        ),
        action=policy_result.action,
        eta_hours=model_output.eta_hours,
        eta_sigma_hours=model_output.eta_sigma_hours,
        p_burn_by_T=PBurnByT.model_validate(model_output.p_burn_by_T) if model_output.p_burn_by_T else None,
        y_hat=model_output.y_hat,
        sigma=model_output.sigma,
        baseline_y=model_output.baseline_y,
        incident=IncidentInfo(
            irwin_id=nearest_incident.irwin_id if nearest_incident else None,
            incident_name=nearest_incident.name if nearest_incident else None,
            # Raw WFIGS values, not the model's zero-filled e_features copy: a genuinely
            # unreported acres/containment must render as null, never a false "0" (FR-40).
            acres=nearest_incident.acres if nearest_incident else None,
            containment_pct=nearest_incident.containment_pct if nearest_incident else None,
            dist_perimeter_m=dist_perim_m,
            hours_since_discovery=e_features.hours_since_discovery,
        ),
        weather=WeatherInfo(
            red_flag=e_features.rflag,
            wind_speed_ms=e_features.wind_speed_ms,
            wind_dir_cardinal=hrrr_weather.wind_dir_cardinal if hrrr_weather else None,
            rh_pct=e_features.rh_pct,
            hrrr_valid_time=hrrr_weather.hrrr_valid_time if hrrr_weather else None,
        ),
        recommended_actions=RECOMMENDED_ACTIONS.get(policy_result.action, []),
        reasons=policy_result.reasons,
        flags= [f for f in dict.fromkeys(flags) if f in set(get_args(FlagEnum))],
        citations=citations,
        e_product_times=e_product_times,
        w_sources=w_sources,
        model_version=model_output.model_version,
        spread_field_version=model_output.spread_field_version,
        policy_version=policy_result.policy_version,
    )

    raw_brief = _write_llm_brief(card)
    brief, validation = safe_brief(raw_brief, card)
    if not validation.valid:
        tool_logger.log_event(
            "brief_validation_failed", card_id=card.card_id, offending_numbers=validation.offending_numbers, severity_violation=validation.severity_violation
        )
    card_briefs[card.card_id] = brief  # stash for the caller; see _CARD_BRIEFS docstring

    return card


# ActionCard has no brief field (SRS 4.4 is closed to LLM-written prose in the schema
# itself); the brief travels alongside the card via this small in-memory map, keyed by
# card_id, so ask/watch callers can retrieve it without changing the schema contract.
card_briefs: dict[str, str] = {}


card_thread_ts: dict[str, str] = {}  # card_id -> Slack thread_ts, so the Response Agent can reply into it

site_aois: dict[str, Geometry] = {}  # site_id -> incident AOI for the Response Agent water map


def _try_spread(
    deps: MainAgentDeps,
    site: Site,
    nearest_incident: Optional[WFIGSIncident],
    perimeters: list,
    hrrr_weather: Optional[HRRRWeather],
    flags: list[str],
) -> tuple[SpreadSiteSample | None, Geometry]:
    """Incident-first AOI + spread_run. Failures degrade to point geometry (FR-33)."""
    geom = point_geometry(site.lat, site.lng)
    if nearest_incident is None or not perimeters:
        return None, geom

    matching = [p for p in perimeters if p.irwin_id and p.irwin_id == nearest_incident.irwin_id]
    perimeter = matching[0] if matching else perimeters[0]
    policy_cfg = load_policy_config()
    aoi_cfg = policy_cfg.get("aoi") or {}
    ens_cfg = policy_cfg.get("spread_ensemble") or {}
    wind_u = hrrr_weather.wind_u_10m if hrrr_weather else None
    wind_v = hrrr_weather.wind_v_10m if hrrr_weather else None

    bbox = aoi_bbox_for_fetch(
        perimeter,
        wind_u,
        wind_v,
        base_buffer_km=float(aoi_cfg.get("base_buffer_km", 5)),
        downwind_buffer_km=float(aoi_cfg.get("downwind_buffer_km", 25)),
        upwind_buffer_km=float(aoi_cfg.get("upwind_buffer_km", 2)),
    )
    if bbox is None:
        return None, geom

    try:
        stack = deps.landfire.fetch_aoi(
            *bbox,
            site_id=site.site_id,
            resample_m=int(aoi_cfg.get("resample_m", 90)),
        )
        geom = build_incident_aoi(
            perimeter,
            stack,
            wind_u=wind_u,
            wind_v=wind_v,
            base_buffer_km=float(aoi_cfg.get("base_buffer_km", 5)),
            downwind_buffer_km=float(aoi_cfg.get("downwind_buffer_km", 25)),
            upwind_buffer_km=float(aoi_cfg.get("upwind_buffer_km", 2)),
            max_cells=int(aoi_cfg.get("max_cells", 12000)),
        )
        flags.extend(geom.flags)
        rings = [list(ring) for ring in perimeter.geometry_rings]
        field = spread_run(
            incident_id=nearest_incident.irwin_id or nearest_incident.name,
            landfire=stack,
            aoi=geom,
            weather={
                "wind_u": wind_u or 0.0,
                "wind_v": wind_v or 0.0,
                "rh_pct": hrrr_weather.rh_pct if hrrr_weather else None,
                "temp_c": hrrr_weather.temp_c if hrrr_weather else None,
                "valid_time": hrrr_weather.hrrr_valid_time if hrrr_weather else None,
            },
            perimeter_rings=rings,
            ignition_points=[{"lat": nearest_incident.lat, "lng": nearest_incident.lng}],
            ensemble={
                "n_members": int(ens_cfg.get("n_members", 7)),
                "wind_speed_frac": float(ens_cfg.get("wind_speed_frac", 0.2)),
                "wind_dir_deg": float(ens_cfg.get("wind_dir_deg", 20)),
                "moisture_frac": float(ens_cfg.get("moisture_frac", 0.15)),
            },
            site_id=site.site_id,
        )
        sample = field.sample(site.lat, site.lng)
        site_aois[site.site_id] = geom
        return sample, geom
    except (LandfireRequestFailed, Exception) as exc:
        logger.warning("spread_run/LANDFIRE failed for %s: %s; continuing without arrival field", site.site_id, exc)
        flags.append("degraded")
        return None, geom


def deliver_action_card_and_state(
    deps: MainAgentDeps, site: Site, card: ActionCard, brief: str, now: datetime | None = None
) -> None:
    now = now or datetime.now(timezone.utc)
    receipt = deliver_action_card(card, brief, site.slack_channel)
    card_thread_ts[card.card_id] = receipt.thread_ts
    deliver_cards_by_email(card, brief)
    deps.site_state.update_state(
        site.site_id, card.action, card.model_dump(mode="json"), card.incident.irwin_id, delivered=True, now=now
    )


ESCALATION_ACTIONS = {"protect_asset", "evacuate_site"}


response_cards: dict[str, Any] = {}  # ActionCard.card_id -> ResponseCard, for ask-mode callers


def _trigger_response_agent(deps: MainAgentDeps, site: Site, card: ActionCard, background: bool) -> Any:
    from src.agents.response_agent import run_response_agent

    if background:
        import threading

        threading.Thread(target=run_response_agent, args=(deps, site, card), daemon=True).start()
        return None
    response_card = run_response_agent(deps, site, card)
    response_cards[card.card_id] = response_card
    return response_card


def run_site(deps: MainAgentDeps, site: Site, mode: str = "ask") -> ActionCard:
    """Full pipeline for one site: build, deliver, and (if escalated) trigger the Response
    Agent. Ask mode runs the response agent synchronously so the CLI can print both cards;
    watch mode fires it in a background thread so the poll cycle is not blocked (FR-44)."""
    now = datetime.now(timezone.utc)
    card = build_action_card(deps, site)
    brief = card_briefs.pop(card.card_id, _fallback_brief(card))

    deliver_action_card_and_state(deps, site, card, brief, now=now)

    if card.action in ESCALATION_ACTIONS:
        _trigger_response_agent(deps, site, card, background=(mode == "watch"))

    return card
