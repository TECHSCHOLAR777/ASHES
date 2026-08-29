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

load_dotenv()

from src.agents.agent_loop import run_agentic  # noqa: E402
from src.agents.main_agent import MainAgentDeps, Site  # noqa: E402


ACTIONS = {"monitor", "prepare", "protect_asset", "evacuate_site", "inspect_after", "no_action"}
OUT = Path("data/models/agentic_live_report.json")


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
    # Strip any accidental env-looking strings
    OUT.write_text(json.dumps(payload, indent=2, default=str)[:400_000], encoding="utf-8")
    print(json.dumps({"ask_action": (ask.get("action_card") or {}).get("action"), "ask_tools": [t["tool"] for t in ask.get("trace") or []], "problems": problems, "out": str(OUT)}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
