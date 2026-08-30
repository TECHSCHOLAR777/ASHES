"""Plain-text email delivery of ActionCard + ResponseCard (SRS 4.3, Step 12).

Runs in log-only mode when SMTP_* is unset (see config/README_MISSING.md): the composed
message body is written to the daily JSONL log instead of sent via smtplib.
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from src.logging_ import tool_logger
from src.schemas.action_card import ActionCard
from src.schemas.response_card import ResponseCard


def _smtp_config() -> dict[str, str] | None:
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    from_addr = os.environ.get("SMTP_FROM")
    to_addr = os.environ.get("SMTP_TO")
    if not all([host, user, password, from_addr, to_addr]):
        return None
    return {
        "host": host,
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "from_addr": from_addr,
        "to_addr": to_addr,
    }


def _compose_body(action_card: ActionCard, action_brief: str, response_card: ResponseCard | None, response_brief: str | None) -> str:
    lines = [
        f"ACTION: {action_card.action.upper()} - {action_card.site.name}",
        f"Incident: {action_card.incident.incident_name or 'none'} "
        f"({action_card.incident.acres if action_card.incident.acres is not None else 'n/a'} ac, "
        f"{action_card.incident.containment_pct if action_card.incident.containment_pct is not None else 'n/a'} contained)",
        f"Distance to perimeter: {action_card.incident.dist_perimeter_m if action_card.incident.dist_perimeter_m is not None else 'n/a'} m",
        f"Engine ETA: {action_card.eta_hours if action_card.eta_hours is not None else 'null'} h (sigma={action_card.sigma})",
        "",
        "Reasons:",
        *[f"- {r}" for r in action_card.reasons],
        "",
        "Brief:",
        action_brief,
    ]
    if response_card is not None:
        lines += [
            "",
            "---- RESPONSE DOSSIER ----",
            f"Responsible agency: {response_card.responsible_agency or 'local'}",
            f"Fire station: {response_card.fire_station.name or 'unknown'} "
            f"({response_card.fire_station.distance_m:.0f} m, ETA {response_card.fire_station.eta_minutes_estimate} min)",
            "Water sources:",
            *[
                f"- {w.type} {w.name or ''} ({f'{w.distance_m:.0f} m' if w.distance_m is not None else 'distance unknown'}, {w.availability})"
                for w in response_card.water_sources
            ],
            "Hazmat priority:",
            *[f"- {h.type} {h.name or ''} ({h.distance_m:.0f} m, {h.priority})" for h in response_card.hazmat_sites],
            "",
            "Response brief:",
            response_brief or "",
        ]
    return "\n".join(lines)


def deliver_cards_by_email(
    action_card: ActionCard,
    action_brief: str,
    response_card: ResponseCard | None = None,
    response_brief: str | None = None,
) -> bool:
    """Returns True if actually sent over SMTP, False if it ran in log-only mode."""
    body = _compose_body(action_card, action_brief, response_card, response_brief)
    config = _smtp_config()

    if config is None:
        tool_logger.log_event(
            "email_delivery_log_only", card_id=action_card.card_id, subject=f"{action_card.action}: {action_card.site.name}", body=body
        )
        return False

    msg = EmailMessage()
    msg["Subject"] = f"{action_card.action}: {action_card.site.name}"
    msg["From"] = config["from_addr"]
    msg["To"] = config["to_addr"]
    msg.set_content(body)

    with smtplib.SMTP(config["host"], config["port"]) as server:
        server.starttls()
        server.login(config["user"], config["password"])
        server.send_message(msg)

    tool_logger.log_event("email_delivery_sent", card_id=action_card.card_id, to=config["to_addr"])
    return True
