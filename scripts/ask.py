"""CLI entry point for ask mode (SRS FR-31, Step 10):

    python scripts/ask.py --lat 33.98 --lng -117.37 --q "Is the site at risk?"

Runs the full pipeline once for the given coordinate and prints the ActionCard (and, if the
action escalates to protect_asset/evacuate_site, the ResponseCard) to stdout. The question
text is accepted for interface completeness (SRS FR-1) but resolution is by coordinate: the
system computes distance/scores from data, never lets the LLM answer the question directly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.agents.main_agent import MainAgentDeps, Site, response_cards, run_site  # noqa: E402
from src.logging_ import tool_logger  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the wildfire copilot about one site")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lng", type=float, required=True)
    parser.add_argument("--q", type=str, default="Is this site at risk?")
    parser.add_argument("--name", type=str, default="ask-mode-site")
    parser.add_argument("--slack-channel", type=str, default="#fire-alerts")
    parser.add_argument("--agentic", action="store_true", help="OpenAI tool-calling loop over live tools")
    parser.add_argument("--simulate", action="store_true", help="Run ELMFIRE from an ignition at lat/lng")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    site = Site(site_id=f"ask_{uuid.uuid4().hex[:8]}", name=args.name, lat=args.lat, lng=args.lng, slack_channel=args.slack_channel)

    deps = MainAgentDeps()
    if args.agentic or args.simulate:
        from src.agents.agent_loop import run_agentic

        report = run_agentic(
            deps,
            site,
            args.q,
            simulate=args.simulate,
            ignition_lat=args.lat if args.simulate else None,
            ignition_lng=args.lng if args.simulate else None,
        )
        print(json.dumps(report, indent=2, default=str))
        tool_logger.log_event(
            "ask_mode_query",
            question=args.q,
            site_id=site.site_id,
            action=(report.get("action_card") or {}).get("action"),
            agentic=True,
            simulate=args.simulate,
        )
        return

    card = run_site(deps, site, mode="ask")

    print("=== ActionCard ===")
    print(card.model_dump_json(indent=2, by_alias=True))

    response_card = response_cards.pop(card.card_id, None)
    if response_card is not None:
        print("\n=== ResponseCard ===")
        print(response_card.model_dump_json(indent=2))

    tool_logger.log_event(
        "ask_mode_query",
        question=args.q,
        site_id=site.site_id,
        action=card.action,
        card_id=card.card_id,
        response_card_id=response_card.card_id if response_card else None,
    )


if __name__ == "__main__":
    main()
