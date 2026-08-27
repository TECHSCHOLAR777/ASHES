"""Mireye API client: round-robin key rotation, sliding-window rate limiting, quote-before-fetch.

Critical module (SRS 1 / FR-15 / rule 5 in the build prompt). Three keys give an effective
180 rpm budget when Mireye rate-limits each key to 60 rpm. `/v1/ask` is banned outright:
Mireye is a physical-facts source fetched by typed field, never asked as a question.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.logging_ import tool_logger

MIREYE_BASE_URL = "https://api.mireye.com"
# The SRS's "60 rpm/key" is a generic figure; the three keys actually issued for this build
# are on Mireye's Growth plan ($99/mo, 120,000 credits, 300 rpm - confirmed by the user
# 2026-08-27), so the limiter uses the real per-key cap. Three keys round-robin to an
# effective ~900 rpm budget.
RATE_LIMIT_PER_KEY = 300
RATE_LIMIT_SAFETY_MARGIN = 1  # skip a key at cap-1 rather than waiting for a 429
RATE_WINDOW_SECONDS = 60.0
MAX_BATCH_SIZE = 25
MAX_RETRIES = 3
# Verified against the live API 2026-08-27: an explicit field list over 50 is rejected with
# {"error": "fields_too_many", "max": 50} (presets are exempt, but V1 never uses a preset).
MAX_FIELDS_PER_REQUEST = 50

# Geocode accuracy_type values observed from the live provider (geocodio) that mean the
# match is a precise point, not an interpolated/approximate one.
GEOCODE_PRECISE_ACCURACY_TYPES = {"rooftop", "point"}


class MireyeAskBanned(ValueError):
    """Raised when a caller attempts to use /v1/ask for a physical fact (SRS 2.6.3)."""


class MireyeRequestFailed(RuntimeError):
    """Raised after retries are exhausted on a Mireye call."""


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


@dataclass
class GeoPoint:
    lat: float
    lng: float
    confidence: str
    range_interpolation: bool


@dataclass
class QuoteResult:
    credits: float
    fields: list[str]
    lat: float
    lng: float


@dataclass
class _KeyWindow:
    timestamps: deque = field(default_factory=deque)


class MireyeClient:
    """Round-robin client over N Mireye API keys with a per-key sliding-window rate limit."""

    def __init__(self, keys: list[str], base_url: str = MIREYE_BASE_URL, timeout: float = 30.0):
        if not keys:
            raise ValueError("MireyeClient requires at least one API key")
        self._keys = list(keys)
        self._base_url = base_url
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._windows: dict[int, _KeyWindow] = {i: _KeyWindow() for i in range(len(self._keys))}
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # -- response shaping ---------------------------------------------------

    @staticmethod
    def _flatten_fields(fields_response: dict[str, Any], fetched_at: str | None) -> dict[str, Any]:
        """Flattens the live API's nested `{"fields": {name: {value, confidence, ...}}}`
        shape into the flat `field`/`field_confidence`/`field_source_url`/`field_vintage`
        convention the rest of the codebase (w_encoder, response_agent) expects."""
        flat: dict[str, Any] = {}
        for name, meta in fields_response.items():
            flat[name] = meta.get("value")
            if meta.get("confidence") is not None:
                flat[f"{name}_confidence"] = meta["confidence"]
            if meta.get("source_url") is not None:
                flat[f"{name}_source_url"] = meta["source_url"]
            if meta.get("dataset_vintage") is not None:
                flat[f"{name}_vintage"] = meta["dataset_vintage"]
        if fetched_at is not None:
            flat["fetched_at"] = fetched_at
        return flat

    # -- key rotation -----------------------------------------------------

    def _prune(self, window: _KeyWindow, now: float) -> None:
        while window.timestamps and now - window.timestamps[0] > RATE_WINDOW_SECONDS:
            window.timestamps.popleft()

    def _pick_key_index(self) -> int:
        """Round-robin starting point, skipping any key at its 60 rpm sliding-window cap."""
        n = len(self._keys)
        with self._lock:
            start = next(self._counter) % n
            now = time.monotonic()
            for offset in range(n):
                idx = (start + offset) % n
                window = self._windows[idx]
                self._prune(window, now)
                if len(window.timestamps) < RATE_LIMIT_PER_KEY - RATE_LIMIT_SAFETY_MARGIN:
                    window.timestamps.append(now)
                    return idx
            # All keys saturated: fall back to the round-robin start and accept the request
            # (Mireye's own 429 handling + our retry/backoff covers the rare overrun).
            window = self._windows[start]
            window.timestamps.append(now)
            return start

    # -- HTTP with retry ----------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None,
        site_id: str | None = None,
    ) -> dict[str, Any]:
        if path.rstrip("/").endswith("/v1/ask"):
            raise MireyeAskBanned(
                "MireyeClient forbids /v1/ask for physical facts; fetch a typed field instead"
            )

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            key_index = self._pick_key_index()
            headers = {"Authorization": f"Bearer {self._keys[key_index]}"}
            start = time.monotonic()
            try:
                resp = self._client.request(method, path, json=json_body, headers=headers)
            except httpx.HTTPError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                tool_logger.log_tool_call(
                    f"mireye:{path}", json_body or {}, None, site_id, latency_ms, key_index, error=str(exc)
                )
                last_error = exc
                time.sleep(2**attempt)
                continue

            latency_ms = (time.monotonic() - start) * 1000
            if resp.status_code == 429 or resp.status_code >= 500:
                tool_logger.log_tool_call(
                    f"mireye:{path}",
                    json_body or {},
                    {"status_code": resp.status_code},
                    site_id,
                    latency_ms,
                    key_index,
                    error=f"HTTP {resp.status_code}",
                )
                last_error = MireyeRequestFailed(f"{path} returned HTTP {resp.status_code}")
                time.sleep(2**attempt)
                continue

            resp.raise_for_status()
            data = resp.json()
            tool_logger.log_tool_call(f"mireye:{path}", json_body or {}, data, site_id, latency_ms, key_index)
            return data

        raise MireyeRequestFailed(f"Mireye {path} failed after {MAX_RETRIES} attempts") from last_error

    # -- public API -----------------------------------------------------

    def geocode(self, address: str) -> GeoPoint:
        data = self._request("POST", "/v1/geocode", {"address": address})
        # The live API returns {lat, lng, accuracy, accuracy_type, match_type,
        # normalized_address, provider, source} - no confidence/range_interpolation fields,
        # so those are derived here (documented in DECISIONS.md).
        accuracy_type = data.get("accuracy_type")
        confidence = accuracy_type or "unknown"
        range_interpolation = accuracy_type is not None and accuracy_type not in GEOCODE_PRECISE_ACCURACY_TYPES
        return GeoPoint(
            lat=data["lat"],
            lng=data["lng"],
            confidence=confidence,
            range_interpolation=range_interpolation,
        )

    def quote(self, lat: float, lng: float, fields: list[str], site_id: str | None = None) -> QuoteResult:
        """Quotes a single point. Chunks internally at MAX_FIELDS_PER_REQUEST and sums cost,
        so a caller never has to know about the live API's 50-field cap."""
        total_credits = 0.0
        for chunk in _chunked(fields, MAX_FIELDS_PER_REQUEST):
            data = self._request(
                "POST", "/v1/fetch/quote", {"lat": lat, "lng": lng, "fields": chunk}, site_id=site_id
            )
            total_credits += float(data.get("credits_total", 0.0))
        tool_logger.log_credit_usage(site_id, quoted_credits=total_credits, actual_credits=None, key_index=None)
        return QuoteResult(credits=total_credits, fields=fields, lat=lat, lng=lng)

    def fetch(self, lat: float, lng: float, fields: list[str], site_id: str | None = None) -> dict[str, Any]:
        """Quote first (SRS FR-15 / rule 5), then fetch. Always logs the quoted cost.

        The live API charges by field-count deterministically (`credits_per_location *
        field_count`), and does not echo an "actual" cost distinct from the quote in the
        fetch response itself, so the logged actual equals the quote (SRS's "quote for
        reproducibility, not rationing" - there is no drift to reconcile here).
        """
        quote_result = self.quote(lat, lng, fields, site_id=site_id)
        merged: dict[str, Any] = {}
        for chunk in _chunked(fields, MAX_FIELDS_PER_REQUEST):
            data = self._request("POST", "/v1/fetch", {"lat": lat, "lng": lng, "fields": chunk}, site_id=site_id)
            merged.update(self._flatten_fields(data.get("fields", {}), data.get("fetched_at")))
        tool_logger.log_credit_usage(site_id, quote_result.credits, quote_result.credits, key_index=None)
        return merged

    def fetch_batch(
        self,
        coords: list[tuple[float, float]],
        fields: list[str],
        site_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Batches coordinates in groups of <= MAX_BATCH_SIZE, quoting each batch before fetch.

        The live batch-quote endpoint takes a location *count*, not the coordinates
        themselves (quoting is location-agnostic; cost only depends on how many points and
        fields are requested).
        """
        results: list[dict[str, Any]] = []
        for start in range(0, len(coords), MAX_BATCH_SIZE):
            chunk = coords[start : start + MAX_BATCH_SIZE]
            batch_label = f"batch_{start}"

            quoted_credits = 0.0
            for field_chunk in _chunked(fields, MAX_FIELDS_PER_REQUEST):
                quote_data = self._request(
                    "POST",
                    "/v1/fetch/quote",
                    {"locations": len(chunk), "fields": field_chunk},
                    site_id=batch_label,
                )
                quoted_credits += float(quote_data.get("credits_total", 0.0))
            tool_logger.log_credit_usage(batch_label, quoted_credits, None, key_index=None)

            merged_by_index: list[dict[str, Any]] = [dict() for _ in chunk]
            for field_chunk in _chunked(fields, MAX_FIELDS_PER_REQUEST):
                data = self._request(
                    "POST",
                    "/v1/fetch/batch",
                    {"locations": [{"lat": lat, "lng": lng} for lat, lng in chunk], "fields": field_chunk},
                    site_id=batch_label,
                )
                for result in data.get("results", []):
                    idx = result["index"]
                    merged_by_index[idx].update(
                        self._flatten_fields(result.get("fields", {}), result.get("fetched_at"))
                    )
            tool_logger.log_credit_usage(batch_label, quoted_credits, quoted_credits, key_index=None)
            results.extend(merged_by_index)
        return results
