"""Main entry point: starts the APScheduler watch loop over the book of sites (SRS FR-31, Step 10).

    python scripts/watch_loop.py

Polls every 5 minutes (config/policy.yaml: poll_interval_minutes) and runs one full cycle
across config/sites.yaml on start, then on every subsequent tick. Ctrl+C to stop.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from apscheduler.schedulers.background import BackgroundScheduler  # noqa: E402

from src.agents.main_agent import MainAgentDeps, load_sites  # noqa: E402
from src.agents.watch_runner import run_watch_cycle  # noqa: E402
from src.policy.engine import load_policy_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fire_copilot.watch_loop")


def main() -> None:
    sites = load_sites()
    deps = MainAgentDeps()
    policy_cfg = load_policy_config()
    poll_interval_minutes = policy_cfg["poll_interval_minutes"]

    logger.info("Starting watch loop over %d sites, poll interval %d min", len(sites), poll_interval_minutes)

    def tick() -> None:
        logger.info("Poll cycle starting")
        run_watch_cycle(deps, sites)
        logger.info("Poll cycle complete")

    # `next_run_time` is intentionally left at its default here: passing `None` explicitly
    # (as an earlier version of this file did) tells APScheduler the job starts paused with
    # no automatic runs at all, not "schedule normally" - confirmed live, it silently
    # produced exactly one poll cycle (the manual `tick()` below) and then nothing ever
    # again (see DECISIONS.md). Leaving the argument out lets APScheduler compute the
    # normal recurring schedule (first automatic fire at now + interval).
    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, "interval", minutes=poll_interval_minutes)
    scheduler.start()

    tick()  # also run once immediately, rather than waiting a full interval for the first card

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down watch loop")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
