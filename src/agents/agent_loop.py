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
from src.agents.main_agent import Site

logger = logging.getLogger("fire_copilot.agent_loop")

SYSTEM_PROMPT = """You are ASHES, an unattended cited wildfire copilot for a book of named US sites.

You actually call tools. You never invent a number, distance, acreage, ETA, or score.
You never decide the Action enum. After gathering facts you MUST call apply_policy, then commit_report.

What each source is for:
- NWS / FIRMS / WFIGS / HRRR: live fire week (E).
- Mireye fields: cited site aspects (fuels, terrain including aspect_degrees/aspect_cardinal, hazard zone, buildings, roads, water, surface_management_agency). Not P(this pixel burns in 72 h).
- spread_run / simulate_ignition: the out-of-process engine (ELMFIRE when configured). ETA and P(burn by T) come from the engine sample, not from you.
- model_infer: combines E + encoded W + engine sample. y_hat is not the action.
- apply_policy: versioned table. First match wins. You may not override it.

You MUST call mireye_fetch (roles A–I at minimum) before commit_report. mireye_quote is not a fetch.
A sealed report without cited site aspects is incomplete.

Tool order for a live ask (call independent live feeds in one turn when you can):
geocode if needed → nws_alerts, firms_hotspots, wfigs_incidents, wfigs_perimeters, hrrr_weather in parallel → list_mireye_roles then mireye_quote + mireye_fetch for roles A–I (add J if you will build a dossier) → run_spread if a WFIGS perimeter exists, else simulate_ignition when the user asked to simulate or there is no live fire → model_infer → apply_policy → build_response_dossier if action is protect_asset or evacuate_site → commit_report.

If run_spread fails, continue degraded and still apply_policy. Never fabricate an ETA.

Copy rules: after commit_report the system writes the brief from grounded JSON. Your job is the tools.
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


def _backfill(session: AgentSession) -> None:
    """If the model skipped required live tools, run them. Honest: labeled backfill in the trace."""
    if session.simulate and "simulate_ignition" not in session.called:
        execute_tool(session, "simulate_ignition", {}, backfill=True)
    for name in BACKFILL_ORDER:
        if name in session.called:
            continue
        if session.simulate and name in {"wfigs_incidents", "wfigs_perimeters"}:
            continue
        args: dict[str, Any] = {}
        if name == "mireye_fetch":
            args = {"roles": ["A", "B", "C", "D", "E", "F", "G", "H", "I"]}
        execute_tool(session, name, args, backfill=True)
    if session.simulate:
        if "simulate_ignition" not in session.called:
            execute_tool(session, "simulate_ignition", {}, backfill=True)
    elif "run_spread" not in session.called:
        execute_tool(session, "run_spread", {}, backfill=True)
    if "model_infer" not in session.called:
        execute_tool(session, "model_infer", {}, backfill=True)
    if "apply_policy" not in session.called:
        execute_tool(session, "apply_policy", {}, backfill=True)
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
    on_event=None,
    client=None,
) -> dict[str, Any]:
    """Drive tools via OpenAI function calling. Falls back to the deterministic tool order if no key."""
    session = AgentSession(
        deps=deps,
        site=site,
        question=question,
        simulate=simulate,
        ignition_lat=ignition_lat,
        ignition_lng=ignition_lng,
        on_event=on_event,
    )
    user_blob = {
        "site": {"site_id": site.site_id, "name": site.name, "lat": site.lat, "lng": site.lng, "address": site.address},
        "question": question,
        "simulate": simulate,
        "ignition": {"lat": ignition_lat, "lng": ignition_lng} if ignition_lat is not None else None,
        "instruction": "Call tools. Finish with apply_policy then commit_report.",
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
