"""Append-only JSONL logging of every tool call, model call, policy call, and credit spend.

Logs are evidence (SRS FR-41, NFR-14): a card must be fully reconstructable from these files.
One file per UTC day at data/logs/YYYY-MM-DD.jsonl. Lines are appended, never rewritten.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"
_LOCK = threading.Lock()


def _log_path(now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"


def _append(record: dict[str, Any]) -> None:
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    line = json.dumps(record, default=str, sort_keys=True)
    with _LOCK:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_tool_call(
    tool_name: str,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    site_id: str | None,
    latency_ms: float,
    key_index: int | None = None,
    error: str | None = None,
) -> None:
    """Log one external tool call. Never pass the raw API key in `request`."""
    _append(
        {
            "kind": "tool_call",
            "tool": tool_name,
            "site_id": site_id,
            "req": request,
            "resp": response,
            "latency_ms": latency_ms,
            "key_idx": key_index,
            "error": error,
        }
    )


def log_model_call(
    site_id: str,
    w_features_shape: tuple,
    e_features_shape: tuple,
    y_hat: float,
    sigma: float,
    baseline_y: float,
    model_version: str,
    eta_hours: float | None = None,
    spread_field_version: str | None = None,
) -> None:
    _append(
        {
            "kind": "model_call",
            "site_id": site_id,
            "w_shape": list(w_features_shape),
            "e_shape": list(e_features_shape),
            "y_hat": y_hat,
            "sigma": sigma,
            "baseline_y": baseline_y,
            "model_version": model_version,
            "eta_hours": eta_hours,
            "spread_field_version": spread_field_version,
        }
    )


def log_policy_call(
    site_id: str,
    y_hat: float | None,
    sigma: float,
    action: str,
    reasons: list[str],
    policy_version: str,
) -> None:
    _append(
        {
            "kind": "policy_call",
            "site_id": site_id,
            "y_hat": y_hat,
            "sigma": sigma,
            "action": action,
            "reasons": reasons,
            "policy_version": policy_version,
        }
    )


def log_credit_usage(
    site_id: str | None,
    quoted_credits: float,
    actual_credits: float | None,
    key_index: int | None,
) -> None:
    _append(
        {
            "kind": "credit_usage",
            "site_id": site_id,
            "quoted_credits": quoted_credits,
            "actual_credits": actual_credits,
            "key_idx": key_index,
        }
    )


def log_event(kind: str, **fields: Any) -> None:
    """Generic structured event log for cases not covered by the typed helpers above."""
    record = {"kind": kind}
    record.update(fields)
    _append(record)
