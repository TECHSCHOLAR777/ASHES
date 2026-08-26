"""SQLite W cache: role-dependent TTL for Mireye fields (SRS FR-16, Step 4).

Caching exists for freshness/latency/reproducibility, never to survive a credit cap
(SRS header + NFR-6). Static roles (B, D, E, F, G, H, I, J) get a long 30-day TTL.
Dynamic roles (A: ndvi_current/ndvi_change_5y; C: drought_category) get a short TTL that
tightens during fire season (May-Oct) because those fields actually move within a season.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "w_cache.db"

STATIC_ROLES = {"B", "D", "E", "F", "G", "H", "I", "J"}
DYNAMIC_ROLES = {"A", "C"}

STATIC_TTL_DAYS = 30
DYNAMIC_TTL_FIRE_SEASON_DAYS = 3
DYNAMIC_TTL_OFF_SEASON_DAYS = 14
FIRE_SEASON_MONTHS = {5, 6, 7, 8, 9, 10}  # May-Oct


def _is_fire_season(now: datetime) -> bool:
    return now.month in FIRE_SEASON_MONTHS


def role_ttl(role: str, now: datetime | None = None) -> timedelta:
    now = now or datetime.now(timezone.utc)
    if role in DYNAMIC_ROLES:
        days = DYNAMIC_TTL_FIRE_SEASON_DAYS if _is_fire_season(now) else DYNAMIC_TTL_OFF_SEASON_DAYS
        return timedelta(days=days)
    return timedelta(days=STATIC_TTL_DAYS)


@dataclass
class CachedW:
    site_id: str
    lat: float
    lng: float
    role: str
    fields: dict[str, Any]
    vintages: dict[str, Any]
    fetched_at: str
    expires_at: str


class WCache:
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
                CREATE TABLE IF NOT EXISTS w_cache (
                    site_id TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lng REAL NOT NULL,
                    field_set_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    vintages_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (site_id, role, field_set_hash)
                )
                """
            )

    def put(
        self,
        site_id: str,
        role: str,
        fields: dict[str, Any],
        vintages: dict[str, Any],
        lat: float = 0.0,
        lng: float = 0.0,
        field_set_hash: str = "default",
        now: datetime | None = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)
        expires_at = now + role_ttl(role, now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO w_cache
                    (site_id, lat, lng, field_set_hash, role, fields_json, vintages_json, fetched_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(site_id, role, field_set_hash) DO UPDATE SET
                    lat=excluded.lat, lng=excluded.lng, fields_json=excluded.fields_json,
                    vintages_json=excluded.vintages_json, fetched_at=excluded.fetched_at,
                    expires_at=excluded.expires_at
                """,
                (
                    site_id,
                    lat,
                    lng,
                    field_set_hash,
                    role,
                    json.dumps(fields),
                    json.dumps(vintages),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )

    def get(
        self, site_id: str, role: str, field_set_hash: str = "default", now: datetime | None = None
    ) -> CachedW | None:
        now = now or datetime.now(timezone.utc)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM w_cache WHERE site_id = ? AND role = ? AND field_set_hash = ?",
                (site_id, role, field_set_hash),
            ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if now > expires_at:
            return None
        return CachedW(
            site_id=row["site_id"],
            lat=row["lat"],
            lng=row["lng"],
            role=row["role"],
            fields=json.loads(row["fields_json"]),
            vintages=json.loads(row["vintages_json"]),
            fetched_at=row["fetched_at"],
            expires_at=row["expires_at"],
        )

    def invalidate(self, site_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM w_cache WHERE site_id = ?", (site_id,))
