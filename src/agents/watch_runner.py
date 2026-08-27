"""Watch-mode orchestration: polling cadence, idempotency, and de-duplication (SRS FR-31/FR-43).

Kept separate from `main_agent.py` so the per-site pipeline (`build_action_card`) and the
scheduling/dedup policy around it are independently testable.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.agents.main_agent import (
    MainAgentDeps,
    Site,
    build_action_card,
    card_briefs,
    deliver_action_card_and_state,
)
from src.agents.main_agent import ESCALATION_ACTIONS, _fallback_brief, _trigger_response_agent
from src.policy.engine import load_policy_config

logger = logging.getLogger("fire_copilot.watch")


def poll_site(deps: MainAgentDeps, site: Site, policy_cfg: dict) -> None:
    now = datetime.now(timezone.utc)
    idempotency_seconds = policy_cfg["poll_idempotency_minutes"] * 60
    dedup_seconds = policy_cfg["dedup_window_minutes"] * 60

    seconds_since_poll = deps.site_state.seconds_since_last_poll(site.site_id, now)
    if seconds_since_poll is not None and seconds_since_poll < idempotency_seconds:
        logger.info("Skipping %s: polled %.0fs ago (idempotency window)", site.site_id, seconds_since_poll)
        return

    try:
        card = build_action_card(deps, site)
    except Exception as exc:
        logger.error("build_action_card failed for %s: %s", site.site_id, exc)
        return
    brief = card_briefs.pop(card.card_id, _fallback_brief(card))

    prior_state = deps.site_state.get_state(site.site_id)
    state_changed = prior_state is None or prior_state.last_action != card.action
    seconds_since_delivery = deps.site_state.seconds_since_last_delivery(site.site_id, now)
    within_dedup_window = seconds_since_delivery is not None and seconds_since_delivery < dedup_seconds

    should_deliver = state_changed or not within_dedup_window
    if should_deliver:
        deliver_action_card_and_state(deps, site, card, brief, now=now)
    else:
        logger.info("De-duplicating %s: same action %s within dedup window", site.site_id, card.action)
        deps.site_state.update_state(
            site.site_id, card.action, card.model_dump(mode="json"), card.incident.irwin_id, delivered=False, now=now
        )

    if should_deliver and card.action in ESCALATION_ACTIONS:
        _trigger_response_agent(deps, site, card, background=True)


def run_watch_cycle(deps: MainAgentDeps, sites: list[Site]) -> None:
    """One full poll cycle across the book of sites, parallelized (NFR-2)."""
    policy_cfg = load_policy_config()
    max_workers = policy_cfg["max_watch_workers"]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(poll_site, deps, site, policy_cfg) for site in sites]
        for future in futures:
            future.result()  # surface any unexpected exception in the calling thread's logs
