"""Per-site state store (SRS FR-32): last poll, last action, last card, for de-dup and idempotency.

This is the only durable "memory" the agent has. The LLM context is never the source of
truth for any number (FR-32); the watch loop reads and writes this table instead.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "site_state.db"


@dataclass
class SiteState:
    site_id: str
    last_poll_at: str | None
    last_action: str | None
    last_card_json: dict[str, Any] | None
    last_incident_irwin_id: str | None
    last_delivery_at: str | None


class SiteStateStore:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS site_state (
                    site_id TEXT PRIMARY KEY,
                    last_poll_at TEXT,
                    last_action TEXT,
                    last_card_json TEXT,
                    last_incident_irwin_id TEXT,
                    last_delivery_at TEXT
                )
                """
            )

    def get_state(self, site_id: str) -> SiteState | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM site_state WHERE site_id = ?", (site_id,)).fetchone()
        if row is None:
            return None
        return SiteState(
            site_id=row["site_id"],
            last_poll_at=row["last_poll_at"],
            last_action=row["last_action"],
            last_card_json=json.loads(row["last_card_json"]) if row["last_card_json"] else None,
            last_incident_irwin_id=row["last_incident_irwin_id"],
            last_delivery_at=row["last_delivery_at"],
        )

    def update_state(
        self,
        site_id: str,
        action: str,
        card_json: dict[str, Any],
        irwin_id: str | None,
        delivered: bool = False,
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)
        now_iso = now.isoformat()
        existing = self.get_state(site_id)
        delivery_at = now_iso if delivered else (existing.last_delivery_at if existing else None)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO site_state
                    (site_id, last_poll_at, last_action, last_card_json, last_incident_irwin_id, last_delivery_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(site_id) DO UPDATE SET
                    last_poll_at=excluded.last_poll_at, last_action=excluded.last_action,
                    last_card_json=excluded.last_card_json,
                    last_incident_irwin_id=excluded.last_incident_irwin_id,
                    last_delivery_at=excluded.last_delivery_at
                """,
                (site_id, now_iso, action, json.dumps(card_json), irwin_id, delivery_at),
            )

    def seconds_since_last_poll(self, site_id: str, now: datetime | None = None) -> float | None:
        state = self.get_state(site_id)
        if state is None or state.last_poll_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        last = datetime.fromisoformat(state.last_poll_at)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds()

    def seconds_since_last_delivery(self, site_id: str, now: datetime | None = None) -> float | None:
        state = self.get_state(site_id)
        if state is None or state.last_delivery_at is None:
            return None
        now = now or datetime.now(timezone.utc)
        last = datetime.fromisoformat(state.last_delivery_at)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (now - last).total_seconds()
