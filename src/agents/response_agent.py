"""Response Support Agent (SRS 2.2.5, 4.5, FR-44 through FR-55, Step 11).

Runs in parallel to the main agent whenever a site reaches `protect_asset` or
`evacuate_site`. No trained model: every number here is a Mireye W field, a live USGS
gage reading, or deterministic code. Auto-triggered by `main_agent`/`watch_runner`.
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import yaml

from src.delivery.email_delivery import deliver_cards_by_email
from src.delivery.slack_delivery import deliver_response_card
from src.logging_ import tool_logger
from src.schemas.action_card import ActionCard
from src.schemas.response_card import (
    AccessRoute,
    Airport,
    Comms,
    EnvironmentalConstraint,
    Evacuation,
    FireStation,
    HazmatSite,
    HospitalRef,
    ResponseCard,
    SiteRef,
    Structure,
    USGSGaugeSummary,
    WaterSource,
)

if TYPE_CHECKING:
    from src.agents.main_agent import MainAgentDeps, Site

logger = logging.getLogger("fire_copilot.response_agent")

RESPONSE_CARD_VERSION = "v1.0.0"

RESPONSE_BRIEF_SYSTEM_PROMPT = (
    "You are a tactical fire-response briefer. You receive a fully populated ResponseCard "
    "JSON describing water sources, access routes, hazmat priorities, and other on-the-ground "
    "resources. Write a 2-3 paragraph plain-English tactical summary that copies the values "
    "verbatim. Do not invent any new numbers, distances, or names. Return only the prose."
)

# Overture Transportation road classes, verified against the live Mireye API 2026-08-27
# (see DECISIONS.md) - not the generic interstate/highway/arterial guess this originally
# shipped with.
ROAD_CLASS_PRIORITY = {
    "motorway": 9, "trunk": 8, "primary": 7, "secondary": 6, "tertiary": 5,
    "residential": 4, "living_street": 3, "unclassified": 2, "service": 1,
    "track": 0, "unknown": 0,
}
SURFACE_PRIORITY = {"paved": 2, "unpaved": 1, "unknown": 0}


def _load_policy_cfg() -> dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "policy.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch_response_w(deps: "MainAgentDeps", site: "Site") -> dict[str, Any]:
    """Reads roles A-I from cache (already fetched by the main pipeline) and fetches role J
    on demand if not yet cached (FR-45)."""
    raw_w: dict[str, Any] = {}
    for role in sorted(deps.field_catalog["roles"].keys()):
        if role == "J":
            continue
        cached = deps.w_cache.get(site.site_id, role)
        if cached is not None:
            raw_w.update(cached.fields)

    cached_j = deps.w_cache.get(site.site_id, "J")
    if cached_j is not None:
        raw_w.update(cached_j.fields)
    else:
        j_fields = list(deps.field_catalog["roles"]["J"]["fields"].keys())
        try:
            fetched = deps.mireye.fetch(site.lat, site.lng, j_fields, site_id=site.site_id)
            raw_w.update(fetched)
            deps.w_cache.put(site.site_id, "J", {k: fetched.get(k) for k in j_fields}, {}, site.lat, site.lng)
        except Exception as exc:
            logger.warning("Role J Mireye fetch failed for %s: %s; response card will be sparse", site.site_id, exc)

    return raw_w


def _water_availability(discharge_cfs: float | None, drought_category: str | None, permanence_pct: float | None, within_service_area: bool | None) -> tuple[str, str | None]:
    note = None
    if drought_category in ("D2", "D3", "D4"):
        note = f"drought {drought_category} - expect low flow"

    if discharge_cfs is not None:
        if discharge_cfs < 10:
            avail = "dry"
        elif discharge_cfs < 50:
            avail = "low"
        elif discharge_cfs < 500:
            avail = "moderate"
        else:
            avail = "high"
    elif permanence_pct is not None:
        avail = "moderate" if permanence_pct > 0.5 else "low"
    elif within_service_area:
        avail = "high"
    else:
        avail = "unknown"

    return avail, note


def _build_water_sources(deps: "MainAgentDeps", site: "Site", raw_w: dict[str, Any], now_iso: str) -> list[WaterSource]:
    gage_id = raw_w.get("nearest_usgs_gage_id")
    gage = deps.usgs.get_gage_discharge(gage_id, site_id=site.site_id)
    drought = raw_w.get("drought_category")
    permanence = raw_w.get("surface_water_permanence_pct")
    within_service = raw_w.get("within_water_service_area")

    sources: list[WaterSource] = []

    if raw_w.get("nearest_flowline_name") or gage_id:
        avail, note = _water_availability(gage.discharge_cfs, drought, permanence, within_service)
        sources.append(
            WaterSource(
                type="stream",
                name=raw_w.get("nearest_flowline_name"),
                distance_m=None,  # Mireye has no flowline distance field, only the name
                discharge_cfs=gage.discharge_cfs,
                permanence_pct=permanence,
                availability=avail,
                note=note,
                source_url=raw_w.get("nearest_flowline_name_source_url", "https://waterservices.usgs.gov/nwis/iv/"),
                fetched_at=now_iso,
            )
        )

    if raw_w.get("nearest_waterbody_name"):
        avail, note = _water_availability(None, drought, permanence, within_service)
        sources.append(
            WaterSource(
                type="lake_reservoir",
                name=raw_w.get("nearest_waterbody_name"),
                distance_m=None,  # Mireye has no waterbody distance field, only the name
                discharge_cfs=None,
                permanence_pct=permanence,
                availability=avail,
                note=note,
                source_url="https://mireye.com",
                fetched_at=now_iso,
            )
        )

    if raw_w.get("wetlands_within_100m_count", 0):
        avail, note = _water_availability(None, drought, permanence, within_service)
        sources.append(
            WaterSource(
                type="wetland",
                name=f"{raw_w.get('wetlands_within_100m_count')} wetland(s) within 100m ({raw_w.get('wetland_acres', 0)} ac)",
                distance_m=100.0,
                discharge_cfs=None,
                permanence_pct=permanence,
                availability=avail,
                note=note,
                source_url="https://mireye.com",
                fetched_at=now_iso,
            )
        )

    if within_service:
        sources.append(
            WaterSource(
                type="municipal_water",
                name="municipal water service area",
                distance_m=None,  # within_water_service_area is boolean; no distance field
                discharge_cfs=None,
                permanence_pct=None,
                availability="high",
                note=None,
                source_url="https://mireye.com",
                fetched_at=now_iso,
            )
        )

    if raw_w.get("nearest_wastewater_plant_name"):
        sources.append(
            WaterSource(
                type="wastewater_plant",
                name=raw_w.get("nearest_wastewater_plant_name"),
                distance_m=float(raw_w.get("nearest_wastewater_plant_distance_m", 0.0) or 0.0),
                discharge_cfs=None,
                permanence_pct=None,
                availability="unknown",
                note=None,
                source_url="https://mireye.com",
                fetched_at=now_iso,
            )
        )

    if raw_w.get("high_hazard_dams_within_10km", 0):
        sources.append(
            WaterSource(
                type="dam_reservoir",
                name=f"{raw_w.get('high_hazard_dams_within_10km')} high-hazard dam(s) within 10km",
                distance_m=10_000.0,
                discharge_cfs=None,
                permanence_pct=None,
                availability="unknown",
                note=None,
                source_url="https://mireye.com",
                fetched_at=now_iso,
            )
        )

    return sources


def _build_access_routes(raw_w: dict[str, Any]) -> list[AccessRoute]:
    routes = []
    if raw_w.get("nearest_major_road_name"):
        routes.append(
            AccessRoute(
                road_name=raw_w.get("nearest_major_road_name"),
                road_class=raw_w.get("nearest_major_road_class", "unknown"),
                surface=raw_w.get("nearest_road_surface", "unknown"),
                distance_m=float(raw_w.get("nearest_major_road_distance_m", 0.0) or 0.0),
                usability="excellent",
            )
        )
    if raw_w.get("nearest_road_distance_m") is not None:
        routes.append(
            AccessRoute(
                road_name=None,
                road_class=raw_w.get("nearest_road_class", "unknown"),
                surface=raw_w.get("nearest_road_surface", "unknown"),
                distance_m=float(raw_w.get("nearest_road_distance_m", 0.0) or 0.0),
                usability="good",
            )
        )

    def sort_key(route: AccessRoute) -> tuple[int, int]:
        return (
            ROAD_CLASS_PRIORITY.get(route.road_class, 0),
            SURFACE_PRIORITY.get(route.surface, 0),
        )

    routes.sort(key=sort_key, reverse=True)
    for i, route in enumerate(routes):
        if i == 0 and route.road_class in ("motorway", "trunk", "primary") and route.surface == "paved":
            route.usability = "excellent"
        elif route.road_class in ("unclassified", "unknown") or route.surface == "unpaved":
            route.usability = "limited"
        else:
            route.usability = "good"
    return routes


def _fire_station_eta(raw_w: dict[str, Any], policy_cfg: dict[str, Any]) -> FireStation:
    distance_m = raw_w.get("nearest_fire_station_distance_m")
    if distance_m is None:
        return FireStation(name=raw_w.get("nearest_fire_station_name"), distance_m=0.0, eta_minutes_estimate=None)

    speed_kmh = (
        policy_cfg["fire_station_eta"]["speed_paved_kmh"]
        if raw_w.get("nearest_road_surface") == "paved"
        else policy_cfg["fire_station_eta"]["speed_unpaved_kmh"]
    )
    eta_minutes = (float(distance_m) / 1000.0) / speed_kmh * 60.0
    return FireStation(name=raw_w.get("nearest_fire_station_name"), distance_m=float(distance_m), eta_minutes_estimate=eta_minutes)


def _hazmat_priority(distance_m: float, tiers: dict[str, float]) -> str:
    if distance_m < tiers["critical"]:
        return "critical"
    if distance_m < tiers["high"]:
        return "high"
    return "medium"


def _build_hazmat_sites(raw_w: dict[str, Any], policy_cfg: dict[str, Any]) -> list[HazmatSite]:
    tiers = policy_cfg["hazmat_priority_tiers_m"]
    candidates: list[tuple[str, str | None, float | None]] = [
        ("EPA_RMP", raw_w.get("nearest_hazardous_facility_name"), raw_w.get("nearest_hazardous_facility_distance_m")),
        ("RCRA_TSD", None, raw_w.get("nearest_rcra_tsd_distance_m")),
        ("UST", None, raw_w.get("nearest_ust_facility_distance_m")),
        ("gas_pipeline", None, raw_w.get("nearest_gas_pipeline_distance_m")),
        ("petroleum_pipeline", None, raw_w.get("nearest_petroleum_pipeline_distance_m")),
        ("transmission_line", None, raw_w.get("nearest_transmission_line_distance_m")),
        ("osm_substation", None, raw_w.get("nearest_osm_substation_distance_m")),
    ]
    sites: list[HazmatSite] = []
    max_radius_m = tiers["medium"]
    for hazmat_type, name, distance_m in candidates:
        if distance_m is None or distance_m > max_radius_m:
            continue
        sites.append(
            HazmatSite(
                type=hazmat_type,
                name=name,
                distance_m=float(distance_m),
                priority=_hazmat_priority(float(distance_m), tiers),
                note=None,
            )
        )
    sites.sort(key=lambda h: h.distance_m)
    return sites


def _build_environmental_constraints(raw_w: dict[str, Any]) -> list[EnvironmentalConstraint]:
    constraints = []
    if raw_w.get("intersects_critical_habitat"):
        constraints.append(
            EnvironmentalConstraint(
                type="critical_habitat",
                species_or_designation=raw_w.get("critical_habitat_species"),
                manager=None,
                constraint="retardant_restricted",
            )
        )
    if raw_w.get("intersects_protected_area"):
        constraints.append(
            EnvironmentalConstraint(
                type="protected_area",
                species_or_designation=raw_w.get("protected_area_designation"),
                manager=raw_w.get("protected_area_manager"),
                constraint="coordinate_first",
            )
        )
    return constraints


def _write_response_brief(card: ResponseCard) -> str:
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        return _fallback_response_brief(card)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RESPONSE_BRIEF_SYSTEM_PROMPT},
                {"role": "user", "content": card.model_dump_json()},
            ],
            max_tokens=400,
        )
        return resp.choices[0].message.content or _fallback_response_brief(card)
    except Exception as exc:
        logger.warning("OpenAI response brief generation failed: %s; using fallback brief", exc)
        return _fallback_response_brief(card)


def _fallback_response_brief(card: ResponseCard) -> str:
    return (
        f"Response dossier for {card.site.name}: {len(card.water_sources)} water source(s), "
        f"{len(card.access_routes)} access route(s), {len(card.hazmat_sites)} hazmat site(s) "
        f"within radius. Responsible agency: {card.responsible_agency or 'local'}. "
        f"Fire station ETA: {card.fire_station.eta_minutes_estimate} minutes."
    )


def run_response_agent(deps: "MainAgentDeps", site: "Site", action_card: ActionCard) -> ResponseCard:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    policy_cfg = _load_policy_cfg()

    raw_w = _fetch_response_w(deps, site)

    water_sources = _build_water_sources(deps, site, raw_w, now_iso)
    access_routes = _build_access_routes(raw_w)
    fire_station = _fire_station_eta(raw_w, policy_cfg)
    airport = Airport(name=raw_w.get("nearest_airport_name"), distance_m=float(raw_w.get("nearest_airport_distance_m", 0.0) or 0.0))
    hazmat_sites = _build_hazmat_sites(raw_w, policy_cfg)
    environmental_constraints = _build_environmental_constraints(raw_w)
    responsible_agency = raw_w.get("surface_management_agency") or "local"
    comms = Comms(
        mobile_5g_coverage=raw_w.get("mobile_5g_coverage_class"),
        nearest_antenna_distance_m=raw_w.get("nearest_antenna_structure_distance_m"),
        nearest_antenna_height_m=raw_w.get("nearest_antenna_structure_height_m"),
        fiber_available=None,
    )
    evacuation = Evacuation(
        housing_units_within_1km=raw_w.get("housing_units_within_1km"),
        housing_density_per_km2=raw_w.get("housing_units_density_per_km2"),
        nearest_hospital=HospitalRef(name=None, distance_m=raw_w.get("nearest_hospital_distance_m")),
        nearest_school_distance_m=raw_w.get("nearest_school_distance_m"),
    )
    structure = Structure(
        height_m=raw_w.get("primary_building_height_m"),
        footprint_sqm=raw_w.get("primary_building_footprint_sqm"),
        overture_class=raw_w.get("primary_building_overture_class"),
    )
    gage_id = raw_w.get("nearest_usgs_gage_id")
    gage = deps.usgs.get_gage_discharge(gage_id, site_id=site.site_id)
    usgs_summary = USGSGaugeSummary(
        gage_name=gage_id, distance_m=None, discharge_cfs=gage.discharge_cfs, discharge_class=gage.discharge_class, fetched_at=gage.fetched_at
    )

    citations = [
        {"source": "mireye", "url": raw_w.get(f"{k}_source_url"), "fetched_at": now_iso, "field": k}
        for k in raw_w
        if k.endswith("_source_url") is False and raw_w.get(f"{k}_source_url")
    ]
    w_sources = [
        {"field": k, "source_url": raw_w.get(f"{k}_source_url", "n/a"), "vintage": raw_w.get(f"{k}_vintage"), "confidence": raw_w.get(f"{k}_confidence")}
        for k in raw_w
        if not k.endswith(("_source_url", "_vintage", "_confidence")) and raw_w.get(f"{k}_confidence") is not None
    ]

    card = ResponseCard(
        card_id=str(uuid.uuid4()),
        triggered_by_action_card=action_card.card_id,
        generated_at=now_iso,
        site=SiteRef(site_id=site.site_id, name=site.name, lat=site.lat, lng=site.lng),
        water_sources=water_sources,
        access_routes=access_routes,
        fire_station=fire_station,
        nearest_airport=airport,
        hazmat_sites=hazmat_sites,
        environmental_constraints=environmental_constraints,
        responsible_agency=responsible_agency,
        comms=comms,
        evacuation=evacuation,
        structure=structure,
        usgs_gage_summary=usgs_summary,
        citations=citations,
        w_sources=w_sources,
        response_card_version=RESPONSE_CARD_VERSION,
    )

    brief = _write_response_brief(card)
    tool_logger.log_event("response_card_generated", card_id=card.card_id, site_id=site.site_id)

    from src.agents.main_agent import card_thread_ts

    thread_ts = card_thread_ts.get(action_card.card_id, f"synthetic-{action_card.card_id}")
    deliver_response_card(card, brief, site.slack_channel, thread_ts)
    deliver_cards_by_email(action_card, "", card, brief)

    return card
