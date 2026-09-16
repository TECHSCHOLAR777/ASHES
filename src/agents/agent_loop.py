"""OpenAI tool-calling loop. The model chooses tools; policy still owns the Action enum."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agents.agent_session import AgentSession
from src.agents.agent_tools import (
    OPENAI_TOOLS,
    PARALLEL_OK,
    execute_tool,
    serialize_report,
)
from src.agents.bucket_skills import BUCKET_SKILLS, situation_for_llm
from src.agents.main_agent import Site

logger = logging.getLogger("fire_copilot.agent_loop")


def _valid_us_point(lat: float | None, lng: float | None) -> bool:
    if lat is None or lng is None:
        return False
    try:
        lat_f, lng_f = float(lat), float(lng)
    except (TypeError, ValueError):
        return False
    if abs(lat_f) < 1e-6 and abs(lng_f) < 1e-6:
        return False
    return 24.0 <= lat_f <= 50.0 and -126.0 <= lng_f <= -66.0

SYSTEM_PROMPT = """You are ASHES, an unattended cited wildfire copilot for a book of named US sites.

You actually call tools. You never invent a number, distance, acreage, ETA, or score.
You never decide the Action enum. After gathering facts you MUST call apply_policy, then the
bucket skill for that action, then commit_report.

What each source is for:
- parse_place: NLP. Town/forest names become a Mireye geocode. Copy the honesty string (point vs town boundary).
- NWS / FIRMS / WFIGS / HRRR: live fire week (E).
- Mireye fields: cited site aspects (fuels, terrain including aspect_degrees/aspect_cardinal, hazard zone, buildings, roads, water, surface_management_agency). Not P(this pixel burns in 72 h). These fields are also given to you after policy in a situation JSON so you can tailor the playbook.
- spread_run / simulate_ignition: the out-of-process engine (ELMFIRE when configured). ETA and P(burn by T) come from the engine sample. There is no tabular hit-model. Do not call model_infer; it does not exist.
- apply_policy: versioned table. First match wins. Distance, Red Flag, FIRMS, engine ETA, Mireye guards. You may not override it.
- Bucket skills (draft_watch_note, stage_prepare_plan, list_suppression_steps, list_evac_steps, inspect_checklist, notify_ops): tailor contact/stop-spread steps from Mireye + engine. Do not invent hydrants or phone numbers.

You MUST call mireye_fetch (roles A–I at minimum) before commit_report. mireye_quote is not a fetch.
A sealed report without cited site aspects is incomplete.

Tool order:
parse_place first → nws_alerts, firms_hotspots, wfigs_incidents, wfigs_perimeters, hrrr_weather in parallel → list_mireye_roles then mireye_quote + mireye_fetch for roles A–I (add J if you will build a dossier) → run_spread if a WFIGS perimeter exists, else simulate_ignition when the user asked to simulate or there is no live fire → apply_policy → the bucket skill(s) named in the situation JSON → commit_report.

If run_spread fails, continue degraded and still apply_policy. Never fabricate an ETA.

After commit_report the system writes the playbook from grounded JSON (parse honesty + Mireye + engine + steps). Your job is the tools.
"""

MAX_ROUNDS = 18
BACKFILL_ORDER = [
    "nws_alerts",
    "firms_hotspots",
    "wfigs_incidents",
    "wfigs_perimeters",
    "hrrr_weather",
    "mireye_fetch",
]


def _openai_client():
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _tool_call_payload(tc: Any) -> tuple[str, str, dict[str, Any]]:
    name = tc.function.name
    raw_args = tc.function.arguments or "{}"
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return tc.id, name, args


def _run_tool_batch(session: AgentSession, calls: list[tuple[str, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Execute one model turn of tool calls. Independent live feeds may run in parallel."""
    messages: list[dict[str, Any]] = []
    if calls and all(name in PARALLEL_OK for _, name, _ in calls):
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(6, len(calls))) as pool:
            futs = {pool.submit(execute_tool, session, name, args): (tc_id, name) for tc_id, name, args in calls}
            for fut in as_completed(futs):
                tc_id, name = futs[fut]
                try:
                    results[tc_id] = fut.result()
                except Exception as exc:
                    results[tc_id] = {"ok": False, "error": str(exc)}
        for tc_id, name, _ in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": json.dumps(results.get(tc_id, {"ok": False}), default=str)[:8000],
                }
            )
        return messages

    for tc_id, name, args in calls:
        result = execute_tool(session, name, args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(result, default=str)[:8000],
            }
        )
    return messages


def _inject_situation(session: AgentSession, messages: list[dict[str, Any]]) -> None:
    if session.policy_result is None or session.situation_injected:
        return
    session.situation_injected = True
    action = session.policy_result.action
    blob = {
        "situation": situation_for_llm(session),
        "instruction": (
            f"Policy action is {action}. Call {BUCKET_SKILLS.get(action, [])} to tailor the "
            "playbook from Mireye aspects + engine sample, then commit_report. "
            "Copy parse.honesty into the playbook. Do not change the Action enum. "
            "Mireye is cited site fact, not P(this pixel burns)."
        ),
    }
    messages.append({"role": "user", "content": json.dumps(blob, default=str)[:24000]})


def _backfill_bucket(session: AgentSession) -> None:
    if session.policy_result is None:
        return
    action = session.policy_result.action
    for name in BUCKET_SKILLS.get(action, []):
        if name == "notify_ops":
            continue
        if name == "build_response_dossier":
            continue
        if name not in session.called:
            execute_tool(session, name, {}, backfill=True)
            break


def _backfill(session: AgentSession) -> None:
    """If the model skipped required live tools, run them. Honest: labeled backfill in the trace."""
    if "parse_place" not in session.called:
        execute_tool(session, "parse_place", {}, backfill=True)
    for name in BACKFILL_ORDER:
        if name in session.called:
            continue
        if session.simulate and name in {"wfigs_incidents", "wfigs_perimeters"}:
            continue
        args: dict[str, Any] = {}
        if name == "mireye_fetch":
            args = {"roles": ["A", "B", "C", "D", "E", "F", "G", "H", "I"]}
        execute_tool(session, name, args, backfill=True)
    want_sim = session.simulate or (
        session.ignition_lat is not None and session.ignition_lng is not None
    )
    if want_sim:
        if "simulate_ignition" not in session.called:
            sim_args: dict[str, Any] = {}
            if session.ignition_lat is not None and session.ignition_lng is not None:
                sim_args = {
                    "lat": session.ignition_lat,
                    "lng": session.ignition_lng,
                    "buffer_km": session.buffer_km or 8,
                }
            execute_tool(session, "simulate_ignition", sim_args, backfill=True)
    elif "run_spread" not in session.called:
        execute_tool(session, "run_spread", {}, backfill=True)
    if "apply_policy" not in session.called:
        execute_tool(session, "apply_policy", {}, backfill=True)
    _backfill_bucket(session)
    if not session.committed:
        execute_tool(session, "commit_report", {}, backfill=True)


def run_agentic(
    deps,
    site: Site,
    question: str,
    *,
    simulate: bool = False,
    ignition_lat: float | None = None,
    ignition_lng: float | None = None,
    buffer_km: float | None = None,
    coords_supplied: bool = True,
    on_event=None,
    client=None,
) -> dict[str, Any]:
    """Drive tools via OpenAI function calling. Falls back to the deterministic tool order if no key."""
    if ignition_lat is not None and ignition_lng is not None:
        simulate = True
    session = AgentSession(
        deps=deps,
        site=site,
        question=question,
        simulate=simulate,
        ignition_lat=ignition_lat,
        ignition_lng=ignition_lng,
        buffer_km=buffer_km,
        coords_supplied=coords_supplied,
        on_event=on_event,
    )
    execute_tool(session, "parse_place", {})
    if not _valid_us_point(session.site.lat, session.site.lng):
        session.emit(
            {
                "event": "error",
                "message": "Could not resolve a US place (refusing 0,0). Name a town or supply lat/lng.",
            }
        )
        report = serialize_report(session)
        session.emit({"event": "complete", "report": report})
        return report
    user_blob = {
        "site": {"site_id": site.site_id, "name": site.name, "lat": site.lat, "lng": site.lng, "address": site.address},
        "question": question,
        "simulate": session.simulate,
        "coords_supplied": coords_supplied,
        "parse": session.parse,
        "ignition": {"lat": ignition_lat, "lng": ignition_lng} if ignition_lat is not None else None,
        "buffer_km": buffer_km or (8 if simulate else None),
        "instruction": (
            "parse_place already ran. Call live E + mireye_fetch + spread, then apply_policy, "
            "bucket skill, commit_report. Copy parse.honesty. "
            "If ignition is set, call simulate_ignition at that chaparral pin."
        ),
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_blob)},
    ]
    openai_client = client if client is not None else _openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    if openai_client is None:
        session.emit({"event": "status", "message": "No OPENAI_KEY; running deterministic tool order"})
        _backfill(session)
        report = serialize_report(session)
        session.emit({"event": "complete", "report": report})
        return report

    try:
        for _round in range(MAX_ROUNDS):
            if session.committed:
                break
            session.emit({"event": "status", "message": f"model turn {_round + 1}"})
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            choice = resp.choices[0].message
            tool_calls = getattr(choice, "tool_calls", None) or []
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": choice.content or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)
            if not tool_calls:
                break
            parsed = [_tool_call_payload(tc) for tc in tool_calls]
            messages.extend(_run_tool_batch(session, parsed))
            _inject_situation(session, messages)
    except Exception as exc:
        logger.warning("agentic loop failed: %s; backfilling remaining tools", exc)
        session.emit({"event": "status", "message": f"model error; backfilling ({exc})"})

    if any(name not in session.called for name in ("nws_alerts", "mireye_fetch")):
        session.committed = False
    if not session.committed:
        _backfill(session)

    report = serialize_report(session)
    session.emit({"event": "complete", "report": report})
    return report
