"""OpenAI tool schemas and handlers. Every number the agent uses comes from these calls."""
from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, get_args

from src.agents.agent_session import AgentSession, ToolTrace
from src.agents.main_agent import (
    ESCALATION_ACTIONS,
    RECOMMENDED_ACTIONS,
    _nearest_incident_and_perimeter_distance,
    card_briefs,
    response_cards,
    site_aois,
)
from src.agents.spread_viz import downsample_field, sample_dict
from src.clients.wfigs import WFIGSIncident, WFIGSPerimeter
from src.features.e_packer import pack_e
from src.features.ros_ellipse import compute_ros_ellipse_feature
from src.features.w_encoder import encode_w, load_field_catalog
from src.geometry.aoi import aoi_bbox_for_fetch, build_incident_aoi, point_geometry
from src.agents.bucket_skills import (
    BUCKET_SKILLS,
    handle_evac_steps,
    handle_inspect,
    handle_notify_ops,
    handle_prepare_plan,
    handle_suppression_steps,
    handle_watch_note,
)
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
from src.spread.client import spread_run
from src.validator.brief_validator import safe_brief

MAX_TOOL_RESULT_CHARS = 8000

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "parse_place",
            "description": (
                "NLP: extract a US town/forest/address from the question and geocode via Mireye "
                "when lat/lng were not supplied. Always call first. Copy the honesty string into "
                "the playbook. Does not invent coordinates."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "Geocode a US address to lat/lng via Mireye. Skip if lat/lng already known.",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nws_alerts",
            "description": "Fetch NWS CAP alerts at the site and SPC day-1 fire-weather outlook. Red Flag is never suppressed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "firms_hotspots",
            "description": "NASA FIRMS thermal hotspots in 5/10/20 km rings around the site.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wfigs_incidents",
            "description": "Live WFIGS incident locations near the site.",
            "parameters": {
                "type": "object",
                "properties": {"radius_km": {"type": "number", "default": 100}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wfigs_perimeters",
            "description": "Live WFIGS operational perimeters (unofficial) and geodesic distance from the site.",
            "parameters": {
                "type": "object",
                "properties": {"radius_km": {"type": "number", "default": 100}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hrrr_weather",
            "description": "Live HRRR 10 m wind, 2 m temp, and RH at the site. This is the only live weather source.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mireye_roles",
            "description": "List Mireye field catalog roles A–J and field names. Use before mireye_fetch.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mireye_quote",
            "description": "Quote Mireye credits for a field list or roles at the site. Always quote before a large fetch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "roles": {"type": "array", "items": {"type": "string"}},
                    "fields": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mireye_fetch",
            "description": (
                "Fetch cited Mireye site facts (fuels, terrain/aspect, hazard, buildings, "
                "roads, water, agency). These are site aspects, not P(the fire hits this pixel)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Role letters A–J. Default A–I (site aspects). Include J for response water/habitat.",
                    },
                    "fields": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_spread",
            "description": (
                "Run the out-of-process spread engine (ELMFIRE when configured) on the nearest "
                "WFIGS incident AOI. Returns site ETA, ensemble sigma, P(burn by T), engine id. "
                "There is no tabular hit-model. y_hat is not used."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_ignition",
            "description": (
                "ELMFIRE simulator: ignite at a lat/lng (default site), fetch LANDFIRE, run spread_run. "
                "Not an official perimeter. Use when the user asked to simulate or there is no live WFIGS fire. "
                "If an ignition pin is provided (west of the community), ignite there — not on the town pin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lng": {"type": "number"},
                    "buffer_km": {"type": "number", "default": 4},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_policy",
            "description": (
                "REQUIRED. Deterministic policy table picks monitor/prepare/protect_asset/"
                "evacuate_site/inspect_after/no_action from distance, Red Flag, FIRMS, "
                "engine ETA, and Mireye guards (density, roads, NDVI, land use). "
                "You never pick the action."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_watch_note",
            "description": "monitor/no_action skill. Tailor a watch note from Mireye agency + engine ETA-null honesty.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stage_prepare_plan",
            "description": "prepare skill. Stage contacts and egress from Mireye roads/agency.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_suppression_steps",
            "description": "protect_asset skill. Contact agency, roads, water from Mireye + engine ETA. Do not invent hydrants.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_evac_steps",
            "description": "evacuate_site skill. Egress + notify. Do not soften evacuate.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_checklist",
            "description": "inspect_after skill. Post-containment inspection only.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_ops",
            "description": "Send the ActionCard to the site Slack/email channel. Log-only if no token. Never invents a phone number.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_response_dossier",
            "description": "Build the ResponseCard (water, access, hazmat, agency) from Mireye role J + USGS/OSM. Call on protect_asset or evacuate_site.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_report",
            "description": "Assemble the ActionCard, write the cited playbook from Mireye+engine (reasoner, not copy-only), and seal. Call after apply_policy and the bucket skill.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _clip(payload: Any) -> dict[str, Any] | list | str | None:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        import json

        text = json.dumps(payload, default=str)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            return {"truncated": True, "preview": text[:MAX_TOOL_RESULT_CHARS]}
        return payload
    return payload


def _fields_for_roles(roles: list[str] | None, extra: list[str] | None = None) -> list[str]:
    catalog = load_field_catalog()
    names: list[str] = []
    wanted = roles or []
    for role in wanted:
        role_cfg = catalog["roles"].get(role)
        if not role_cfg:
            continue
        names.extend(role_cfg["fields"].keys())
    if extra:
        names.extend(extra)
    # preserve order, drop dupes
    return list(dict.fromkeys(names))


CORE_ASPECT_FIELDS = [
    "aspect_degrees",
    "aspect_cardinal",
    "elevation",
    "surface_management_agency",
    "nearest_road_class",
    "fire_hazard_severity_zone_class",
    "land_use_class",
    "housing_units_density_per_km2",
    "ndvi_current",
]


def default_aspect_roles() -> list[str]:
    return ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


def aspects_by_role(raw_w: dict[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_field_catalog()
    out: dict[str, dict[str, Any]] = {}
    for role, cfg in catalog["roles"].items():
        fields: dict[str, Any] = {}
        for name in cfg["fields"].keys():
            if name not in raw_w:
                continue
            fields[name] = {
                "value": raw_w.get(name),
                "confidence": raw_w.get(f"{name}_confidence"),
                "vintage": raw_w.get(f"{name}_vintage"),
                "source_url": raw_w.get(f"{name}_source_url"),
            }
        if fields:
            out[role] = {"name": cfg.get("name", role), "fields": fields}
    return out


def _incident_json(inc: WFIGSIncident) -> dict[str, Any]:
    return {
        "irwin_id": inc.irwin_id,
        "name": inc.name,
        "acres": inc.acres,
        "containment_pct": inc.containment_pct,
        "lat": inc.lat,
        "lng": inc.lng,
        "discovery_datetime": inc.discovery_datetime,
    }


def _perimeter_rings_json(perimeters: list[WFIGSPerimeter], limit: int = 2) -> list[dict[str, Any]]:
    out = []
    for p in perimeters[:limit]:
        rings = []
        for ring in p.geometry_rings[:1]:
            # downsample vertices for the map
            step = max(1, len(ring) // 200)
            rings.append([[float(lng), float(lat)] for lng, lat in ring[::step]])
        out.append(
            {
                "irwin_id": p.irwin_id,
                "name": p.name,
                "unofficial": True,
                "rings": rings,
            }
        )
    return out


def _hotspot_json(firms: FIRMSResult | None) -> list[dict[str, Any]]:
    if firms is None or firms.unavailable:
        return []
    pts = []
    for hs in (firms.hotspots_20km or [])[:40]:
        pts.append({"lat": hs.lat, "lng": hs.lng, "frp": hs.frp, "source": hs.source})
    return pts


def _synthetic_perimeter(lat: float, lng: float, radius_m: float = 150.0) -> WFIGSPerimeter:
    dlat = radius_m / 111_000.0
    dlng = radius_m / (111_000.0 * max(math.cos(math.radians(lat)), 0.1))
    ring = [
        (lng - dlng, lat - dlat),
        (lng + dlng, lat - dlat),
        (lng + dlng, lat + dlat),
        (lng - dlng, lat + dlat),
        (lng - dlng, lat - dlat),
    ]
    return WFIGSPerimeter(irwin_id="SIMULATED", name="simulated_ignition", geometry_rings=[ring])


def _pack_session(session: AgentSession) -> None:
    site = session.site
    nearest, dist_perim_m = _nearest_incident_and_perimeter_distance(
        site, session.incidents, session.perimeters
    )
    ros_dist = compute_ros_ellipse_feature(
        site_lat=site.lat,
        site_lng=site.lng,
        fire_lat=nearest.lat if nearest else None,
        fire_lng=nearest.lng if nearest else None,
        wind_u=session.hrrr.wind_u_10m if session.hrrr else None,
        wind_v=session.hrrr.wind_v_10m if session.hrrr else None,
        acres=nearest.acres if nearest else None,
        dist_perim_m=dist_perim_m if dist_perim_m is not None else 999_999.0,
    )
    session.e_features = pack_e(
        firms=session.firms,
        wfigs_incident=nearest,
        wfigs_perim_dist_m=dist_perim_m,
        cap_alerts=session.cap_alerts,
        spc=session.spc,
        hrrr=session.hrrr,
        ros_ellipse_dist_m=ros_dist,
    )
    if session.e_features.firms_unavailable:
        session.flags.append("firms_unavailable")
    if session.e_features.perimeter_unofficial:
        session.flags.append("perimeter_unofficial")
    if session.e_features.stale_e:
        session.flags.append("stale_E")
    now = datetime.now(timezone.utc)
    session.w_features = encode_w(session.raw_w, session.deps.field_catalog, now=now)


def _run_engine(
    session: AgentSession,
    incident: WFIGSIncident,
    perimeter: WFIGSPerimeter,
    *,
    base_buffer_km: float,
    downwind_buffer_km: float,
    upwind_buffer_km: float,
    resample_m: int,
    n_members: int,
    reuse: bool = True,
    incident_id: str | None = None,
) -> dict[str, Any]:
    site = session.site
    policy_cfg = load_policy_config()
    aoi_cfg = policy_cfg.get("aoi") or {}
    hrrr = session.hrrr
    wind_u = hrrr.wind_u_10m if hrrr else None
    wind_v = hrrr.wind_v_10m if hrrr else None
    bbox = aoi_bbox_for_fetch(
        perimeter,
        wind_u,
        wind_v,
        base_buffer_km=base_buffer_km,
        downwind_buffer_km=downwind_buffer_km,
        upwind_buffer_km=upwind_buffer_km,
    )
    if bbox is None:
        return {"ok": False, "error": "aoi_bbox_empty"}
    stack = session.deps.landfire.fetch_aoi(
        *bbox,
        site_id=site.site_id,
        resample_m=resample_m,
    )
    geom = build_incident_aoi(
        perimeter,
        stack,
        wind_u=wind_u,
        wind_v=wind_v,
        base_buffer_km=base_buffer_km,
        downwind_buffer_km=downwind_buffer_km,
        upwind_buffer_km=upwind_buffer_km,
        max_cells=int(aoi_cfg.get("max_cells", 12000)),
    )
    session.flags.extend(geom.flags)
    rings = [list(ring) for ring in perimeter.geometry_rings]
    field = spread_run(
        incident_id=incident_id or incident.irwin_id or incident.name,
        landfire=stack,
        aoi=geom,
        weather={
            "wind_u": wind_u or 0.0,
            "wind_v": wind_v or 0.0,
            "rh_pct": hrrr.rh_pct if hrrr else None,
            "temp_c": hrrr.temp_c if hrrr else None,
            "valid_time": hrrr.hrrr_valid_time if hrrr else None,
        },
        perimeter_rings=rings,
        ignition_points=[{"lat": incident.lat, "lng": incident.lng}],
        ensemble={
            "n_members": n_members,
            "wind_speed_frac": 0.2,
            "wind_dir_deg": 20,
            "moisture_frac": 0.15,
        },
        site_id=site.site_id,
        reuse=reuse,
    )
    session.spread_field = field
    session.geom = geom
    session.spread_sample = field.sample(site.lat, site.lng)
    session.spread_viz = downsample_field(field)
    site_aois[site.site_id] = geom
    return {
        "ok": True,
        "engine": field.engine,
        "spread_field_version": field.spread_field_version,
        "n_members": field.n_members,
        "site_sample": sample_dict(session.spread_sample),
        "front": field.nearest_reached(site.lat, site.lng),
        "p_burn_field_max": field.p_burn_field_max(),
        "aoi_flags": geom.flags,
        "grid": list(field.arrival_hours.shape),
        "bounds": {"west": field.west, "south": field.south, "east": field.east, "north": field.north},
    }


def _engine_clock(session: AgentSession) -> dict[str, Any]:
    """ETA / P(burn) from the spread sample. No h_fire y_hat."""
    sample = session.spread_sample
    eta = sample.eta_hours if sample else None
    eta_sigma = sample.eta_sigma_hours if sample else None
    sigma = float(eta_sigma) if eta_sigma is not None else 0.0
    p_burn = None
    if sample is not None:
        p_burn = {
            "24": sample.p_burn_24,
            "48": sample.p_burn_48,
            "72": sample.p_burn_72,
        }
    version = None
    if sample is not None:
        version = sample.spread_field_version
    elif session.spread_field is not None:
        version = session.spread_field.spread_field_version
    return {
        "eta_hours": eta,
        "eta_sigma_hours": eta_sigma,
        "sigma": sigma,
        "p_burn_by_T": p_burn,
        "spread_field_version": version,
        "y_hat": None,
        "model_version": "none",
        "baseline_y": None,
    }


def _assemble_card(session: AgentSession) -> ActionCard:
    if session.e_features is None or session.w_features is None:
        _pack_session(session)
    if session.policy_result is None:
        handle_apply_policy(session, {})
    assert session.e_features is not None
    assert session.w_features is not None
    assert session.policy_result is not None

    site = session.site
    now = datetime.now(timezone.utc)
    nearest, dist_perim_m = _nearest_incident_and_perimeter_distance(
        site, session.incidents, session.perimeters
    )
    clock = _engine_clock(session)
    policy_result = session.policy_result
    flags = [f for f in dict.fromkeys(session.flags) if f in set(get_args(FlagEnum))]
    citations = [Citation(**c) for c in session.w_features.citations]
    w_sources = [
        WSource(
            field=field_name,
            source_url=session.raw_w.get(f"{field_name}_source_url", "n/a"),
            vintage=str(vintage),
            confidence=session.raw_w.get(f"{field_name}_confidence"),
        )
        for field_name, vintage in session.w_features.vintages.items()
    ]
    e_product_times = [
        EProductTime(feed="nws_cap", product_time=now.isoformat()),
        EProductTime(feed="firms", product_time=now.isoformat()),
        EProductTime(feed="wfigs", product_time=now.isoformat()),
        EProductTime(
            feed="hrrr",
            product_time=session.hrrr.hrrr_valid_time if session.hrrr else now.isoformat(),
        ),
    ]
    geom = session.geom or point_geometry(site.lat, site.lng)
    recs = list(RECOMMENDED_ACTIONS.get(policy_result.action, []))
    if session.playbook_steps:
        recs = list(session.playbook_steps)
    p_burn = PBurnByT.model_validate(clock["p_burn_by_T"]) if clock["p_burn_by_T"] else None
    card = ActionCard(
        card_id=str(uuid.uuid4()),
        generated_at=now.isoformat(),
        site=SiteRef(
            site_id=site.site_id,
            name=site.name,
            lat=site.lat,
            lng=site.lng,
            mode="aoi" if geom.mode == "aoi" and geom.contains_point(site.lat, site.lng) else "point",
            geocode_confidence=session.geocode_confidence,
            range_interpolation=session.range_interpolation,
        ),
        action=policy_result.action,
        eta_hours=clock["eta_hours"],
        eta_sigma_hours=clock["eta_sigma_hours"],
        p_burn_by_T=p_burn,
        y_hat=None,
        sigma=clock["sigma"],
        baseline_y=None,
        incident=IncidentInfo(
            irwin_id=nearest.irwin_id if nearest else None,
            incident_name=nearest.name if nearest else None,
            acres=nearest.acres if nearest else None,
            containment_pct=nearest.containment_pct if nearest else None,
            dist_perimeter_m=dist_perim_m,
            hours_since_discovery=session.e_features.hours_since_discovery,
        ),
        weather=WeatherInfo(
            red_flag=session.e_features.rflag,
            wind_speed_ms=session.e_features.wind_speed_ms,
            wind_dir_cardinal=session.hrrr.wind_dir_cardinal if session.hrrr else None,
            rh_pct=session.e_features.rh_pct,
            hrrr_valid_time=session.hrrr.hrrr_valid_time if session.hrrr else None,
        ),
        recommended_actions=recs,
        reasons=policy_result.reasons,
        flags=flags,
        citations=citations,
        e_product_times=e_product_times,
        w_sources=w_sources,
        model_version="none",
        spread_field_version=clock["spread_field_version"],
        policy_version=policy_result.policy_version,
    )
    session.action_card = card
    return card


def _fallback_playbook(session: AgentSession) -> str:
    card = session.action_card
    assert card is not None
    honesty = (session.parse or {}).get("honesty") or ""
    steps = session.playbook_steps or card.recommended_actions
    step_txt = " ".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) if steps else ""
    eta = "null" if card.eta_hours is None else f"{card.eta_hours}"
    return (
        f"{honesty} Action: {card.action} (policy {card.policy_version}). "
        f"Engine clock ETA={eta} h (sigma={card.sigma}). "
        f"Distance to perimeter: {card.incident.dist_perimeter_m} m. "
        f"Reasons: {'; '.join(card.reasons)}. "
        f"Playbook: {step_txt}".strip()
    )


def _write_grounded_brief(session: AgentSession) -> str:
    import json
    import os

    card = session.action_card
    assert card is not None
    grounded = grounded_payload(session)
    api_key = os.environ.get("OPENAI_KEY")
    prompt = (
        "You are the ASHES playbook reasoner. Policy already picked the Action enum; you may "
        "not change it. You receive grounded JSON: parse honesty, ActionCard, Mireye site "
        "aspects, engine sample, bucket playbook steps, copy_these_numbers. "
        "Write: (1) the parse honesty sentence if present — community pin for the named place; "
        "(2) the action as given; (3) engine ETA and P(burn by T) as the clock; "
        "(4) tailored next steps from the playbook and Mireye facts (agency, roads, water, terrain). "
        "Copy numbers only from copy_these_numbers. If notify.log_only, say the alert "
        "was logged. If action is evacuate_site, lead with that. Return prose."
    )
    if not api_key:
        brief = _fallback_playbook(session)
        session.brief = brief
        session.brief_replaced = False
        card_briefs[card.card_id] = brief
        return brief
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(grounded, default=str)[:24000]},
            ],
            max_tokens=700,
        )
        raw = resp.choices[0].message.content or _fallback_playbook(session)
    except Exception as exc:
        tool_logger.log_tool_call("write_brief", {}, None, session.site.site_id, 0.0, error=str(exc))
        raw = _fallback_playbook(session)
    brief, validation = safe_brief(raw, card, grounded=grounded)
    session.brief_replaced = not validation.valid
    if not validation.valid:
        tool_logger.log_event(
            "brief_validation_failed",
            card_id=card.card_id,
            offending_numbers=validation.offending_numbers,
            severity_violation=validation.severity_violation,
        )
    session.brief = brief
    card_briefs[card.card_id] = brief
    return brief


def copyable_numbers(*objs: Any) -> list[float]:
    """Rounded forms the brief writer may copy so the validator does not suppress 258.85 vs 258.8479."""
    found: list[float] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            v = float(x)
            if not math.isfinite(v):
                return
            found.extend((v, round(v, 1), round(v, 2), float(int(round(v)))))
            return
        if isinstance(x, dict):
            for val in x.values():
                walk(val)
            return
        if isinstance(x, (list, tuple)):
            for val in x:
                walk(val)

    for obj in objs:
        walk(obj)
    uniq = sorted({n for n in found if math.isfinite(n)})
    return uniq[:400]


def grounded_payload(session: AgentSession) -> dict[str, Any]:
    card = session.action_card
    aspects = aspects_by_role(session.raw_w)
    engine = {
        "sample": sample_dict(session.spread_sample),
        "engine": session.spread_field.engine if session.spread_field else None,
        "spread_field_version": session.spread_field.spread_field_version if session.spread_field else None,
    }
    card_dump = card.model_dump(mode="json", by_alias=True) if card else None
    response = session.response_card.model_dump(mode="json") if session.response_card is not None else None
    parse = session.parse or {}
    playbook = session.playbook_steps
    notify = session.notify
    return {
        "action_card": card_dump,
        "aspects": aspects,
        "engine": engine,
        "response_card": response,
        "question": session.question,
        "parse": parse,
        "playbook": playbook,
        "notify": notify,
        "copy_these_numbers": copyable_numbers(card_dump, aspects, engine, response, parse, playbook, notify),
    }


def serialize_report(session: AgentSession) -> dict[str, Any]:
    card = session.action_card
    geom = session.geom
    return {
        "site": {
            "site_id": session.site.site_id,
            "name": session.site.name,
            "lat": session.site.lat,
            "lng": session.site.lng,
        },
        "question": session.question,
        "simulate": session.simulate,
        "action_card": card.model_dump(mode="json", by_alias=True) if card else None,
        "response_card": session.response_card.model_dump(mode="json") if session.response_card is not None else None,
        "brief": session.brief,
        "brief_replaced": session.brief_replaced,
        "aspects": aspects_by_role(session.raw_w),
        "engine": {
            "sample": sample_dict(session.spread_sample),
            "field": session.spread_viz,
            "engine": session.spread_field.engine if session.spread_field else None,
            "n_members": session.spread_field.n_members if session.spread_field else None,
            "front": session.spread_field.nearest_reached(session.site.lat, session.site.lng)
            if session.spread_field
            else None,
            "p_burn_field_max": session.spread_field.p_burn_field_max() if session.spread_field else None,
        },
        "map": {
            "perimeters": _perimeter_rings_json(session.perimeters),
            "hotspots": _hotspot_json(session.firms),
            "aoi": geom.geom if geom is not None else None,
            "ignition": {"lat": session.ignition_lat, "lng": session.ignition_lng}
            if session.ignition_lat is not None
            else None,
        },
        "trace": [
            {
                "tool": t.tool,
                "ok": t.ok,
                "latency_ms": t.latency_ms,
                "backfill": t.backfill,
                "error": t.error,
                "args": t.args,
            }
            for t in session.traces
        ],
        "flags": list(dict.fromkeys(session.flags)),
        "quoted_credits": session.quoted_credits,
        "parse": session.parse,
        "playbook": session.playbook_steps,
        "notify": session.notify,
    }


# -- handlers ----------------------------------------------------------------


def handle_parse_place(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    from src.agents.place_parse import parse_place

    parsed = parse_place(session.deps, session.site, session.question, session.coords_supplied)
    session.parse = parsed
    if parsed.get("geocoded"):
        session.geocode_confidence = str(parsed.get("confidence") or "n/a")
        session.range_interpolation = bool(parsed.get("range_interpolation"))
        if parsed.get("place") and session.site.name in {"ask-mode-site", "Ask", ""}:
            session.site.name = str(parsed["place"])
    if parsed.get("want_simulate"):
        session.simulate = True
    return {"ok": True, **parsed}


def handle_geocode(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    address = args.get("address") or session.site.address
    if not address:
        return {"ok": False, "error": "no_address"}
    geo = session.deps.mireye.geocode(address)
    session.site.lat, session.site.lng = geo.lat, geo.lng
    session.geocode_confidence = geo.confidence
    session.range_interpolation = geo.range_interpolation
    return {"ok": True, "lat": geo.lat, "lng": geo.lng, "confidence": geo.confidence}


def handle_nws(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    site = session.site
    session.cap_alerts = session.deps.nws.get_cap_alerts(site.lat, site.lng, site_id=site.site_id)
    session.spc = session.deps.nws.get_spc_outlook(site_id=site.site_id)
    alerts = [
        {
            "event": a.event,
            "severity": a.severity,
            "is_red_flag": a.is_red_flag,
            "effective": a.effective,
            "expires": a.expires,
        }
        for a in session.cap_alerts
    ]
    return {
        "ok": True,
        "red_flag": any(a.is_red_flag for a in session.cap_alerts),
        "alerts": alerts[:12],
        "spc_elevated": bool(session.spc and session.spc.elevated),
    }


def handle_firms(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    site = session.site
    try:
        session.firms = session.deps.firms.get_hotspots(site.lat, site.lng, 1, site.site_id)
    except Exception as exc:
        session.flags.append("firms_unavailable")
        return {"ok": False, "error": str(exc), "unavailable": True}
    f = session.firms
    return {
        "ok": True,
        "unavailable": f.unavailable,
        "count_5km": f.count_5km,
        "count_10km": f.count_10km,
        "count_20km": f.count_20km,
        "frp_sum_5km": f.frp_sum_5km,
        "hotspots": _hotspot_json(f)[:15],
    }


def handle_incidents(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    if session.simulate and session.incidents and session.incidents[0].irwin_id == "SIMULATED":
        return {"ok": True, "skipped": "simulated_ignition", "count": 1, "incidents": [_incident_json(session.incidents[0])]}
    radius = float(args.get("radius_km") or 100)
    session.incidents = session.deps.wfigs.get_incidents(
        session.site.lat, session.site.lng, radius, session.site.site_id
    )
    nearest, _ = _nearest_incident_and_perimeter_distance(session.site, session.incidents, session.perimeters)
    return {
        "ok": True,
        "count": len(session.incidents),
        "incidents": [_incident_json(i) for i in session.incidents[:8]],
        "nearest": _incident_json(nearest) if nearest else None,
    }


def handle_perimeters(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    if session.simulate and session.perimeters and session.perimeters[0].irwin_id == "SIMULATED":
        nearest, dist = _nearest_incident_and_perimeter_distance(session.site, session.incidents, session.perimeters)
        return {
            "ok": True,
            "skipped": "simulated_ignition",
            "count": 1,
            "dist_perimeter_m": dist,
            "nearest_incident": _incident_json(nearest) if nearest else None,
            "unofficial": True,
        }
    radius = float(args.get("radius_km") or 100)
    session.perimeters = session.deps.wfigs.get_perimeters(
        session.site.lat, session.site.lng, radius, session.site.site_id
    )
    nearest, dist = _nearest_incident_and_perimeter_distance(session.site, session.incidents, session.perimeters)
    return {
        "ok": True,
        "count": len(session.perimeters),
        "dist_perimeter_m": dist,
        "nearest_incident": _incident_json(nearest) if nearest else None,
        "unofficial": True,
    }


def handle_hrrr(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    try:
        session.hrrr = session.deps.hrrr.get_weather(session.site.lat, session.site.lng, session.site.site_id)
    except Exception as exc:
        session.flags.append("stale_E")
        return {"ok": False, "error": str(exc)}
    h = session.hrrr
    return {
        "ok": True,
        "wind_speed_ms": h.wind_speed_ms,
        "wind_dir_cardinal": h.wind_dir_cardinal,
        "temp_c": h.temp_c,
        "rh_pct": h.rh_pct,
        "hrrr_valid_time": h.hrrr_valid_time,
        "stale": h.stale,
    }


def handle_list_roles(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    catalog = load_field_catalog()
    roles = {
        role: {"name": cfg.get("name"), "fields": list(cfg["fields"].keys())}
        for role, cfg in catalog["roles"].items()
    }
    return {"ok": True, "roles": roles}


def handle_mireye_quote(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    roles = args.get("roles") or default_aspect_roles()
    fields = _fields_for_roles(roles, args.get("fields"))
    quote = session.deps.mireye.quote(session.site.lat, session.site.lng, fields, site_id=session.site.site_id)
    session.quoted_credits = quote.credits
    return {"ok": True, "credits": quote.credits, "n_fields": len(fields)}


def handle_mireye_fetch(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    roles = args.get("roles") or default_aspect_roles()
    fields = _fields_for_roles(roles, args.get("fields"))
    if not fields:
        return {"ok": False, "error": "no_fields"}
    from src.clients.mireye import MireyeRequestFailed

    try:
        fetched = session.deps.mireye.fetch(session.site.lat, session.site.lng, fields, site_id=session.site.site_id)
    except MireyeRequestFailed as exc:
        if "402" not in str(exc) and "credits" not in str(exc).lower():
            raise
        fetched: dict[str, Any] = {}
        for name in CORE_ASPECT_FIELDS:
            try:
                one = session.deps.mireye.fetch(
                    session.site.lat, session.site.lng, [name], site_id=session.site.site_id
                )
            except MireyeRequestFailed:
                break
            fetched.update(one)
        if not fetched:
            raise
        session.flags.append("degraded")
    session.raw_w.update(fetched)
    now = datetime.now(timezone.utc)
    catalog = session.deps.field_catalog
    for role in roles:
        role_fields = list(catalog["roles"].get(role, {}).get("fields", {}).keys())
        role_data = {k: fetched.get(k) for k in role_fields if k in fetched}
        vintages = {k: fetched.get(f"{k}_vintage") for k in role_fields if fetched.get(f"{k}_vintage")}
        if role_data:
            session.deps.w_cache.put(
                session.site.site_id, role, role_data, vintages, session.site.lat, session.site.lng, now=now
            )
    if session.raw_w.get("fire_hazard_severity_zone_class") is None:
        session.flags.append("fhsz_missing")
    aspects = aspects_by_role(session.raw_w)
    # compact view for the LLM: values only
    summary = {
        role: {n: meta["value"] for n, meta in block["fields"].items() if meta.get("value") is not None}
        for role, block in aspects.items()
    }
    return {"ok": True, "n_fields": len(fetched), "aspects": summary}


def handle_run_spread(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    nearest, _ = _nearest_incident_and_perimeter_distance(session.site, session.incidents, session.perimeters)
    if nearest is None or not session.perimeters:
        return {"ok": False, "error": "no_wfigs_incident_or_perimeter", "hint": "call simulate_ignition"}
    matching = [p for p in session.perimeters if p.irwin_id and p.irwin_id == nearest.irwin_id]
    perimeter = matching[0] if matching else session.perimeters[0]
    policy_cfg = load_policy_config()
    aoi_cfg = policy_cfg.get("aoi") or {}
    ens_cfg = policy_cfg.get("spread_ensemble") or {}
    try:
        return _run_engine(
            session,
            nearest,
            perimeter,
            base_buffer_km=float(aoi_cfg.get("base_buffer_km", 5)),
            downwind_buffer_km=float(aoi_cfg.get("downwind_buffer_km", 25)),
            upwind_buffer_km=float(aoi_cfg.get("upwind_buffer_km", 2)),
            resample_m=int(aoi_cfg.get("resample_m", 90)),
            n_members=int(ens_cfg.get("n_members", 7)),
        )
    except Exception as exc:
        session.flags.append("degraded")
        return {"ok": False, "error": str(exc)}


def handle_simulate(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    live = [p for p in session.perimeters if p.irwin_id != "SIMULATED"]
    live_inc = [i for i in session.incidents if i.irwin_id != "SIMULATED"]
    if not session.simulate and (live or live_inc):
        return {
            "ok": False,
            "error": "live_perimeter_exists",
            "hint": "call run_spread for the WFIGS incident; simulate_ignition is only for --simulate / no live fire",
        }
    lat = args.get("lat")
    lng = args.get("lng")
    if lat is None:
        lat = session.ignition_lat if session.ignition_lat is not None else session.site.lat
    if lng is None:
        lng = session.ignition_lng if session.ignition_lng is not None else session.site.lng
    lat = float(lat)
    lng = float(lng)
    buffer_km = float(args.get("buffer_km") or session.buffer_km or 8)
    session.simulate = True
    session.ignition_lat = lat
    session.ignition_lng = lng
    perimeter = _synthetic_perimeter(lat, lng)
    incident = WFIGSIncident(
        irwin_id="SIMULATED",
        name="simulated_ignition",
        acres=None,
        containment_pct=None,
        discovery_datetime=datetime.now(timezone.utc).isoformat(),
        lat=lat,
        lng=lng,
    )
    session.incidents = [incident]
    session.perimeters = [perimeter]
    ens_cfg = load_policy_config().get("spread_ensemble") or {}
    try:
        result = _run_engine(
            session,
            incident,
            perimeter,
            base_buffer_km=buffer_km,
            downwind_buffer_km=max(buffer_km, 6.0),
            upwind_buffer_km=1.0,
            resample_m=90,
            n_members=int(ens_cfg.get("n_members", 7)),
            reuse=False,
            incident_id=f"SIMULATED:{uuid.uuid4().hex[:8]}",
        )
        result["simulated"] = True
        result["ignition"] = {"lat": lat, "lng": lng}
        return result
    except Exception as exc:
        session.flags.append("degraded")
        return {"ok": False, "error": str(exc), "simulated": True}


def handle_model_infer(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    """Watch-loop leftover. Not registered on the agentic tool list."""
    _pack_session(session)
    assert session.e_features is not None
    assert session.w_features is not None
    session.model_output = model_infer(
        session.site.site_id,
        session.w_features,
        session.e_features,
        session.w_features.vintages,
        spread=session.spread_sample,
    )
    mo = session.model_output
    tool_logger.log_model_call(
        session.site.site_id,
        session.w_features.vector.shape,
        (15,),
        mo.y_hat,
        mo.sigma,
        mo.baseline_y,
        mo.model_version,
        eta_hours=mo.eta_hours,
        spread_field_version=mo.spread_field_version,
    )
    return {
        "ok": True,
        "y_hat": mo.y_hat,
        "sigma": mo.sigma,
        "baseline_y": mo.baseline_y,
        "eta_hours": mo.eta_hours,
        "eta_sigma_hours": mo.eta_sigma_hours,
        "p_burn_by_T": mo.p_burn_by_T,
        "model_version": mo.model_version,
        "spread_field_version": mo.spread_field_version,
    }


def handle_apply_policy(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    if session.e_features is None:
        _pack_session(session)
    assert session.e_features is not None
    clock = _engine_clock(session)
    prior_state = session.deps.site_state.get_state(session.site.site_id)
    previously_in_play = bool(prior_state and prior_state.last_action not in (None, "no_action", "monitor"))
    road_access_limited = session.raw_w.get("nearest_road_class") in (
        "unclassified",
        "service",
        "track",
        "unknown",
        None,
    )
    pin = PolicyInput(
        dist_perim_m=session.e_features.dist_perim_m,
        red_flag=session.e_features.rflag,
        spc_elevated=session.e_features.spc_day1_elevated,
        firms_count_5km=session.e_features.firms_count_5km,
        perimeter_unofficial=session.e_features.perimeter_unofficial,
        land_use_class=session.raw_w.get("land_use_class"),
        ndvi_current=session.raw_w.get("ndvi_current"),
        housing_density_per_km2=session.raw_w.get("housing_units_density_per_km2"),
        road_access_limited=road_access_limited,
        containment_pct=session.e_features.containment_pct,
        previously_in_play=previously_in_play,
        eta_hours=clock["eta_hours"],
        eta_sigma_hours=clock["eta_sigma_hours"],
    )
    session.policy_result = apply_policy(None, clock["sigma"], pin)
    pr = session.policy_result
    tool_logger.log_policy_call(
        session.site.site_id, None, clock["sigma"], pr.action, pr.reasons, pr.policy_version
    )
    session.flags.extend(pr.flags)
    return {
        "ok": True,
        "action": pr.action,
        "reasons": pr.reasons,
        "policy_version": pr.policy_version,
        "flags": pr.flags,
        "clock": "engine_eta_distance_not_h_fire",
        "bucket_skills": BUCKET_SKILLS.get(pr.action, []),
        "note": "Action is from the policy table, not the LLM.",
    }


def handle_response(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    if session.action_card is None:
        _assemble_card(session)
    from src.agents.response_agent import run_response_agent

    if session.raw_w.get("nearest_waterbody_name") is None and session.raw_w.get("nearest_flowline_name") is None:
        try:
            handle_mireye_fetch(session, {"roles": ["J"]})
        except Exception:
            pass
    card = run_response_agent(session.deps, session.site, session.action_card)
    session.response_card = card
    response_cards[session.action_card.card_id] = card
    return {
        "ok": True,
        "card_id": card.card_id,
        "water_sources": len(card.water_sources),
        "access_routes": len(card.access_routes),
        "hazmat_sites": len(card.hazmat_sites),
        "responsible_agency": card.responsible_agency,
        "fire_station": {
            "name": card.fire_station.name,
            "distance_m": card.fire_station.distance_m,
            "eta_minutes_estimate": card.fire_station.eta_minutes_estimate,
            "eta_source": card.fire_station.eta_source,
        },
    }


def _run_bucket_skill(session: AgentSession, *, backfill: bool = False) -> None:
    if session.policy_result is None:
        return
    skills = [s for s in BUCKET_SKILLS.get(session.policy_result.action, []) if s != "notify_ops"]
    if not session.playbook_steps:
        for name in skills:
            if name not in session.called and name != "build_response_dossier":
                execute_tool(session, name, {}, backfill=backfill)
                break


def handle_commit(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    value_keys = [
        k for k in session.raw_w if not str(k).endswith(("_confidence", "_vintage", "_source_url"))
    ]
    if len(value_keys) < 8:
        execute_tool(session, "mireye_fetch", {"roles": default_aspect_roles()}, backfill=True)
    if session.policy_result is None:
        handle_apply_policy(session, {})
    _run_bucket_skill(session, backfill=True)
    card = _assemble_card(session)
    if session.policy_result and session.policy_result.action in ESCALATION_ACTIONS and session.response_card is None:
        try:
            handle_response(session, {})
        except Exception as exc:
            session.flags.append("degraded")
            tool_logger.log_tool_call("build_response_dossier", {}, None, session.site.site_id, 0.0, error=str(exc))
    brief = _write_grounded_brief(session)
    if "notify_ops" in BUCKET_SKILLS.get(card.action, []) and "notify_ops" not in session.called:
        execute_tool(session, "notify_ops", {}, backfill=True)
    session.committed = True
    return {
        "ok": True,
        "card_id": card.card_id,
        "action": card.action,
        "brief": brief,
        "brief_replaced": session.brief_replaced,
        "eta_hours": card.eta_hours,
        "y_hat": None,
        "sigma": card.sigma,
        "spread_field_version": card.spread_field_version,
        "parse_honesty": (session.parse or {}).get("honesty"),
        "playbook": session.playbook_steps,
    }


HANDLERS = {
    "parse_place": handle_parse_place,
    "geocode": handle_geocode,
    "nws_alerts": handle_nws,
    "firms_hotspots": handle_firms,
    "wfigs_incidents": handle_incidents,
    "wfigs_perimeters": handle_perimeters,
    "hrrr_weather": handle_hrrr,
    "list_mireye_roles": handle_list_roles,
    "mireye_quote": handle_mireye_quote,
    "mireye_fetch": handle_mireye_fetch,
    "run_spread": handle_run_spread,
    "simulate_ignition": handle_simulate,
    "apply_policy": handle_apply_policy,
    "draft_watch_note": handle_watch_note,
    "stage_prepare_plan": handle_prepare_plan,
    "list_suppression_steps": handle_suppression_steps,
    "list_evac_steps": handle_evac_steps,
    "inspect_checklist": handle_inspect,
    "notify_ops": handle_notify_ops,
    "build_response_dossier": handle_response,
    "commit_report": handle_commit,
}

PARALLEL_OK = {
    "nws_alerts",
    "firms_hotspots",
    "wfigs_incidents",
    "wfigs_perimeters",
    "hrrr_weather",
    "list_mireye_roles",
    "mireye_quote",
}


def execute_tool(session: AgentSession, name: str, args: dict[str, Any], *, backfill: bool = False) -> dict[str, Any]:
    handler = HANDLERS.get(name)
    start = time.monotonic()
    if handler is None:
        result = {"ok": False, "error": f"unknown_tool:{name}"}
        latency = (time.monotonic() - start) * 1000
        session.traces.append(ToolTrace(name, False, latency, args, result, result["error"], backfill))
        return result
    try:
        result = handler(session, args or {})
        ok = bool(result.get("ok", True))
        err = None if ok else str(result.get("error"))
    except Exception as exc:
        ok = False
        err = str(exc)
        result = {"ok": False, "error": err}
        if name in {"run_spread", "simulate_ignition", "mireye_fetch", "hrrr_weather", "firms_hotspots"}:
            session.flags.append("degraded")
    latency = (time.monotonic() - start) * 1000
    with session.lock:
        session.called.add(name)
        session.traces.append(
            ToolTrace(name, ok, latency, args or {}, _clip(result) if isinstance(result, dict) else result, err, backfill)
        )
    session.emit(
        {
            "event": "tool",
            "tool": name,
            "ok": ok,
            "latency_ms": latency,
            "backfill": backfill,
            "error": err,
            "result": _clip(result) if isinstance(result, dict) else {"value": str(result)[:500]},
        }
    )
    if ok and name in {"simulate_ignition", "run_spread"} and session.spread_viz:
        session.emit(
            {
                "event": "spread",
                "field": session.spread_viz,
                "site": {"lat": session.site.lat, "lng": session.site.lng, "name": session.site.name},
                "ignition": {"lat": session.ignition_lat, "lng": session.ignition_lng}
                if session.ignition_lat is not None
                else None,
            }
        )
    return result if isinstance(result, dict) else {"ok": ok, "result": result}
