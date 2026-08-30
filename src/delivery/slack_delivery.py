"""Slack delivery of ActionCard + ResponseCard (SRS 4.3, FR-39, FR-54, Step 12).

Runs in log-only mode when SLACK_BOT_TOKEN is unset (see config/README_MISSING.md): the
composed message is written to the daily JSONL log instead of posted, and a synthetic
`thread_ts` (derived from card_id) keeps the ResponseCard threading logic working end to
end without a real Slack workspace.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.logging_ import tool_logger
from src.schemas.action_card import ActionCard
from src.schemas.response_card import ResponseCard

SEVERITY_COLOR = {
    "no_action": "#808080",
    "monitor": "#2E86C1",
    "prepare": "#F1C40F",
    "protect_asset": "#E67E22",
    "evacuate_site": "#C0392B",
    "inspect_after": "#8E44AD",
}


@dataclass
class DeliveryReceipt:
    thread_ts: str
    log_only: bool


def _action_card_blocks(card: ActionCard, brief: str) -> list[dict]:
    top_reasons = card.reasons[:3]
    citation_lines = [f"- <{c.url}|{c.source}>" for c in card.citations[:10]]
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{card.action.upper()} - {card.site.name}"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Incident:* {card.incident.incident_name or 'none'} "
                    f"({card.incident.acres if card.incident.acres is not None else 'n/a'} ac, "
                    f"{card.incident.containment_pct if card.incident.containment_pct is not None else 'n/a'} contained)\n"
                    f"*Distance to perimeter:* {card.incident.dist_perimeter_m if card.incident.dist_perimeter_m is not None else 'n/a'} m\n"
                    f"*Engine ETA:* {card.eta_hours if card.eta_hours is not None else 'null'} h "
                    f"(sigma={card.sigma})"
                    + (f"\n*Model score:* y_hat={card.y_hat}" if card.y_hat is not None else "")
                ),
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Key factors:*\n" + "\n".join(f"- {r}" for r in top_reasons)},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Citations:*\n" + ("\n".join(citation_lines) or "none")},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Brief:*\n{brief}"}},
    ]


def deliver_action_card(card: ActionCard, brief: str, channel: str) -> DeliveryReceipt:
    token = os.environ.get("SLACK_BOT_TOKEN")
    blocks = _action_card_blocks(card, brief)

    if not token:
        tool_logger.log_event(
            "slack_delivery_log_only", card_type="action_card", channel=channel, card_id=card.card_id, blocks=blocks
        )
        return DeliveryReceipt(thread_ts=f"synthetic-{card.card_id}", log_only=True)

    from slack_sdk import WebClient  # imported lazily: only needed when a real token exists

    client = WebClient(token=token)
    resp = client.chat_postMessage(channel=channel, blocks=blocks, text=f"{card.action}: {card.site.name}")
    tool_logger.log_tool_call(
        "slack:post_action_card", {"channel": channel, "card_id": card.card_id}, dict(resp.data), None, 0.0
    )
    return DeliveryReceipt(thread_ts=resp["ts"], log_only=False)


def _response_card_blocks(card: ResponseCard, brief: str) -> list[dict]:
    water_lines = [
        f"- {w.type} {w.name or ''} ({f'{w.distance_m:.0f} m' if w.distance_m is not None else 'distance unknown'}, {w.availability})"
        for w in card.water_sources
    ]
    access_lines = [f"- {a.road_name or 'unnamed'} ({a.road_class}/{a.surface}, {a.distance_m:.0f} m)" for a in card.access_routes]
    hazmat_lines = [f"- {h.type} {h.name or ''} ({h.distance_m:.0f} m, {h.priority})" for h in card.hazmat_sites]
    constraint_lines = [f"- {c.type}: {c.constraint}" for c in card.environmental_constraints]

    return [
        {"type": "header", "text": {"type": "plain_text", "text": f"Response Dossier - {card.site.name}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Water sources:*\n" + ("\n".join(water_lines) or "none within radius")}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Access routes:*\n" + ("\n".join(access_lines) or "none within radius")}},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Fire station:* {card.fire_station.name or 'unknown'} ({card.fire_station.distance_m:.0f} m, ETA {card.fire_station.eta_minutes_estimate} min)",
            },
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Hazmat priority:*\n" + ("\n".join(hazmat_lines) or "none within 5 km")}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Environmental constraints:*\n" + ("\n".join(constraint_lines) or "none")}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Responsible agency:* {card.responsible_agency or 'local'}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Brief:*\n{brief}"}},
    ]


def deliver_response_card(card: ResponseCard, brief: str, channel: str, thread_ts: str) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN")
    blocks = _response_card_blocks(card, brief)

    if not token:
        tool_logger.log_event(
            "slack_delivery_log_only",
            card_type="response_card",
            channel=channel,
            card_id=card.card_id,
            thread_ts=thread_ts,
            blocks=blocks,
        )
        return

    from slack_sdk import WebClient

    client = WebClient(token=token)
    resp = client.chat_postMessage(
        channel=channel, thread_ts=thread_ts, blocks=blocks, text=f"Response Dossier: {card.site.name}"
    )
    tool_logger.log_tool_call(
        "slack:post_response_card", {"channel": channel, "card_id": card.card_id}, dict(resp.data), None, 0.0
    )
