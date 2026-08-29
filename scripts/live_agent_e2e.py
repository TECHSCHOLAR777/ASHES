#!/usr/bin/env python3
"""Live agentic e2e: OpenAI + Mireye + live feeds + optional ELMFIRE simulator.

Writes a JSON report with no secrets. Exit 0 only if an ActionCard and a tool
trace exist and policy produced a closed Action enum.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.agents.agent_loop import run_agentic  # noqa: E402
from src.agents.main_agent import MainAgentDeps, Site  # noqa: E402


ACTIONS = {"monitor", "prepare", "protect_asset", "evacuate_site", "inspect_after", "no_action"}
OUT = Path("data/models/agentic_live_report.json")
SUMMARY = Path("data/models/agentic_live_summary.json")


def _run(simulate: bool) -> dict:
    deps = MainAgentDeps()
    site = Site(
        site_id=f"live_{uuid.uuid4().hex[:8]}",
        name="Riverside Warehouse",
        lat=33.9806,
        lng=-117.3755,
    )
    q = (
        "Simulate ignition at the warehouse and report engine arrival, Mireye aspects, and the policy action."
        if simulate
        else "Is this warehouse at risk this fire week? Use live feeds, Mireye aspects, and the spread engine if a fire is nearby."
    )
    return run_agentic(
        deps,
        site,
        q,
        simulate=simulate,
        ignition_lat=site.lat if simulate else None,
        ignition_lng=site.lng if simulate else None,
    )


def _aspect_values(report: dict) -> dict:
    out = {}
    for role, block in (report.get("aspects") or {}).items():
        fields = block.get("fields") if isinstance(block, dict) else None
        if not fields:
            continue
        vals = {}
        for name, meta in fields.items():
            if isinstance(meta, dict):
                if meta.get("value") is not None:
                    vals[name] = meta["value"]
            elif meta is not None:
                vals[name] = meta
        if vals:
            out[role] = vals
    return out


def _slim(report: dict | None) -> dict | None:
    if not report:
        return None
    card = report.get("action_card") or {}
    eng = report.get("engine") or {}
    sample = eng.get("sample") or {}
    return {
        "action": card.get("action"),
        "incident": (card.get("incident") or {}).get("incident_name"),
        "irwin_id": (card.get("incident") or {}).get("irwin_id"),
        "dist_perimeter_m": (card.get("incident") or {}).get("dist_perimeter_m"),
        "spread_field_version": card.get("spread_field_version"),
        "engine": eng.get("engine"),
        "eta_hours": card.get("eta_hours"),
        "p_burn_by_T": card.get("p_burn_by_T"),
        "site_inside_aoi": sample.get("inside_aoi"),
        "sample": sample,
        "tools": [t.get("tool") for t in report.get("trace") or []],
        "aspects": _aspect_values(report),
        "brief_replaced": report.get("brief_replaced"),
        "flags": card.get("flags"),
        "simulate": report.get("simulate"),
        "brief": (report.get("brief") or "")[:800],
    }


def _ok(report: dict, simulate: bool) -> list[str]:
    problems = []
    card = report.get("action_card")
    if not card:
        problems.append("no ActionCard")
        return problems
    if card.get("action") not in ACTIONS:
        problems.append(f"bad action {card.get('action')}")
    tools = [t["tool"] for t in report.get("trace") or []]
    for required in ("nws_alerts", "mireye_fetch", "apply_policy", "commit_report"):
        if required not in tools:
            problems.append(f"missing tool {required}")
    if not report.get("aspects"):
        problems.append("no Mireye aspects")
    if simulate:
        eng = (report.get("engine") or {}).get("engine")
        if not eng:
            problems.append("simulator produced no engine id (LANDFIRE/spread degraded)")
        if "simulate_ignition" not in tools:
            problems.append("missing tool simulate_ignition")
    return problems


def main() -> int:
    if not os.environ.get("OPENAI_KEY"):
        print("OPENAI_KEY missing", file=sys.stderr)
        return 2
    if not (os.environ.get("MIREYE_KEY_1") or os.environ.get("MIREYE_KEY_2")):
        print("MIREYE_KEY missing", file=sys.stderr)
        return 2

    ask = _run(simulate=False)
    sim_flag = os.environ.get("ASHES_LIVE_SIMULATE", "1").lower() in {"1", "true", "yes"}
    payload = {"ask": ask, "simulate": None}
    problems = _ok(ask, simulate=False)
    if sim_flag:
        try:
            sim = _run(simulate=True)
            payload["simulate"] = sim
            problems.extend(f"simulate: {p}" for p in _ok(sim, simulate=True))
        except Exception as exc:
            payload["simulate_error"] = str(exc)
            problems.append(f"simulate raised: {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str)[:400_000], encoding="utf-8")
    summary = {
        "ask": _slim(ask),
        "simulate": _slim(payload.get("simulate") if isinstance(payload.get("simulate"), dict) else None),
        "simulate_error": payload.get("simulate_error"),
        "problems": problems,
        "out": str(OUT),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"ask_action": (ask.get("action_card") or {}).get("action"), "ask_tools": [t["tool"] for t in ask.get("trace") or []], "simulate_action": ((payload.get("simulate") or {}).get("action_card") or {}).get("action") if isinstance(payload.get("simulate"), dict) else None, "simulate_engine": ((payload.get("simulate") or {}).get("engine") or {}).get("engine") if isinstance(payload.get("simulate"), dict) else None, "problems": problems, "summary": str(SUMMARY)}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
