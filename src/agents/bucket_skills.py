"""Per-bucket skills. Policy picks the Action; these tools tailor the playbook from Mireye + engine."""
from __future__ import annotations

from typing import Any

from src.agents.agent_session import AgentSession
from src.agents.spread_viz import sample_dict


BUCKET_SKILLS: dict[str, list[str]] = {
    "no_action": ["draft_watch_note"],
    "monitor": ["draft_watch_note"],
    "prepare": ["stage_prepare_plan", "notify_ops"],
    "protect_asset": ["list_suppression_steps", "notify_ops", "build_response_dossier"],
    "evacuate_site": ["list_evac_steps", "notify_ops", "build_response_dossier"],
    "inspect_after": ["inspect_checklist"],
}


def situation_for_llm(session: AgentSession) -> dict[str, Any]:
    """Mireye aspects + engine sample go into the API call. Not a hit-model vector."""
    from src.agents.agent_tools import aspects_by_role

    action = session.policy_result.action if session.policy_result else None
    sample = sample_dict(session.spread_sample)
    return {
        "parse": session.parse,
        "action": action,
        "bucket_skills": BUCKET_SKILLS.get(action or "", []),
        "engine": {
            "sample": sample,
            "engine": session.spread_field.engine if session.spread_field else None,
            "spread_field_version": session.spread_field.spread_field_version if session.spread_field else None,
            "simulated": session.simulate,
        },
        "aspects": aspects_by_role(session.raw_w),
        "weather": {
            "red_flag": bool(session.e_features.rflag) if session.e_features else False,
            "wind_speed_ms": session.e_features.wind_speed_ms if session.e_features else None,
            "rh_pct": session.e_features.rh_pct if session.e_features else None,
        }
        if session.e_features
        else None,
        "note": (
            "Mireye fields are cited site facts (agency, roads, water, terrain, buildings). "
            "ETA and P(burn by T) come from ELMFIRE. "
            "You may tailor steps from aspects. You may not change the Action enum."
        ),
    }


def _agency(session: AgentSession) -> str:
    return session.raw_w.get("surface_management_agency") or "unknown (Mireye did not return surface_management_agency)"


def _road(session: AgentSession) -> str:
    return session.raw_w.get("nearest_road_class") or "unknown"


def handle_watch_note(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    sample = sample_dict(session.spread_sample) or {}
    steps = [
        "Keep the watch cadence.",
        f"Responsible land agency (Mireye): {_agency(session)}.",
        "Re-ask if a WFIGS perimeter moves inside the 15 km monitor band or a Red Flag is issued.",
    ]
    if sample.get("eta_hours") is None and session.spread_field:
        steps.insert(0, "Engine field exists; this site sits outside a reached cell, so ETA is blank.")
    session.playbook_steps = steps
    return {"ok": True, "bucket": session.policy_result.action if session.policy_result else "monitor", "steps": steps}


def handle_prepare_plan(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    steps = [
        "Stage documents and a site contact list.",
        f"Confirm road class for egress (Mireye nearest_road_class={_road(session)}).",
        f"Notify {_agency(session)} that the site is in the 24–72 h prepare window if ETA is set.",
        "Confirm water service / nearest flowline from Mireye before any wet-line plan.",
    ]
    session.playbook_steps = steps
    return {"ok": True, "bucket": "prepare", "steps": steps}


def handle_suppression_steps(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    sample = sample_dict(session.spread_sample) or {}
    water = session.raw_w.get("nearest_waterbody_name") or session.raw_w.get("nearest_flowline_name")
    steps = [
        f"ETA {sample.get('eta_hours')} h (engine). Protect the asset.",
        f"Contact {_agency(session)}.",
        f"Access: nearest_road_class={_road(session)}.",
        f"Water (Mireye): {water or 'none fetched.'}",
        "Clear defensible space only if staff can do so before arrival.",
    ]
    session.playbook_steps = steps
    return {"ok": True, "bucket": "protect_asset", "steps": steps, "agency": _agency(session)}


def handle_evac_steps(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    steps = [
        "Policy action is evacuate_site. Lead with that. Do not soften it.",
        f"Egress road class (Mireye): {_road(session)}.",
        f"Notify {_agency(session)} and local emergency services of occupancy status.",
        "Use the response dossier routes.",
    ]
    session.playbook_steps = steps
    return {"ok": True, "bucket": "evacuate_site", "steps": steps}


def handle_inspect(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    steps = [
        "Incident is contained and this site was previously in play.",
        "Schedule a post-incident structural and environmental inspection.",
        "Close any active evacuate or protect order.",
    ]
    session.playbook_steps = steps
    return {"ok": True, "bucket": "inspect_after", "steps": steps}


def handle_notify_ops(session: AgentSession, args: dict[str, Any]) -> dict[str, Any]:
    """Slack/email if configured; otherwise JSONL log-only. Never invents a phone number."""
    from src.delivery.slack_delivery import deliver_action_card

    if session.action_card is None:
        from src.agents.agent_tools import _assemble_card

        _assemble_card(session)
    if session.action_card is None:
        return {"ok": False, "error": "no_action_card", "hint": "call apply_policy then commit or assemble first"}
    channel = session.site.slack_channel or "#fire-alerts"
    receipt = deliver_action_card(session.action_card, session.brief or "", channel)
    session.notify = {"channel": channel, "log_only": receipt.log_only, "thread_ts": receipt.thread_ts}
    return {"ok": True, "log_only": receipt.log_only, "channel": channel, "thread_ts": receipt.thread_ts}
