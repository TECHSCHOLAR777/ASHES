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
from concurrent.futures import ThreadPoolExecutor
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
# 429 is the provider's sliding window, not a transient 5xx. Three 1/2/4s
# retries die on a 60s rpm cap; wait the window instead.
MAX_429_RETRIES = 8
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
        """Round-robin starting point, skipping any key at its sliding-window cap.

        If every key is at cap, wait until the oldest stamp ages out rather than
        firing a request we know will 429. The limiter is in-process only; a
        fresh encoder still needs HTTP 429 backoff after another process just
        spent the same keys.
        """
        n = len(self._keys)
        while True:
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
                wait_s = RATE_WINDOW_SECONDS
                for window in self._windows.values():
                    if window.timestamps:
                        wait_s = min(
                            wait_s,
                            RATE_WINDOW_SECONDS - (now - window.timestamps[0]) + 0.05,
                        )
            time.sleep(max(0.05, wait_s))

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
        generic_attempts = 0
        retries_429 = 0
        retries_402 = 0
        while generic_attempts < MAX_RETRIES:
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
                generic_attempts += 1
                time.sleep(2 ** (generic_attempts - 1))
                continue

            latency_ms = (time.monotonic() - start) * 1000
            if resp.status_code == 429:
                tool_logger.log_tool_call(
                    f"mireye:{path}",
                    json_body or {},
                    {"status_code": 429},
                    site_id,
                    latency_ms,
                    key_index,
                    error="HTTP 429",
                )
                last_error = MireyeRequestFailed(f"{path} returned HTTP 429")
                retries_429 += 1
                if retries_429 >= MAX_429_RETRIES:
                    break
                retry_after = resp.headers.get("Retry-After")
                try:
                    sleep_s = min(120.0, float(retry_after)) if retry_after else None
                except ValueError:
                    sleep_s = None
                if sleep_s is None:
                    sleep_s = min(90.0, 20.0 * (2 ** (retries_429 - 1)))
                time.sleep(sleep_s)
                continue

            if resp.status_code == 402:
                tool_logger.log_tool_call(
                    f"mireye:{path}",
                    json_body or {},
                    {"status_code": 402},
                    site_id,
                    latency_ms,
                    key_index,
                    error="HTTP 402 credits_exhausted",
                )
                last_error = MireyeRequestFailed(f"{path} returned HTTP 402")
                retries_402 += 1
                if retries_402 >= max(len(self._keys), 3):
                    break
                continue

            if resp.status_code >= 500:
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
                generic_attempts += 1
                time.sleep(2 ** (generic_attempts - 1))
                continue

            if resp.status_code >= 400:
                # A 4xx is a client-side error (bad field name, too many fields, ...) that
                # will not fix itself on retry, but it must still be logged - it previously
                # wasn't: `raise_for_status()` raised before the log call ever ran, so a
                # sample-building or W-fetch failure's actual response body (e.g.
                # `fields_unknown`) was silently lost, violating NFR-14's "every tool call
                # is reconstructable from logs." Confirmed against a live 400 that vanished
                # from the JSONL log entirely before this fix (see DECISIONS.md).
                body_preview = resp.text[:500]
                tool_logger.log_tool_call(
                    f"mireye:{path}",
                    json_body or {},
                    {"status_code": resp.status_code, "body": body_preview},
                    site_id,
                    latency_ms,
                    key_index,
                    error=f"HTTP {resp.status_code}",
                )
                raise MireyeRequestFailed(f"{path} returned HTTP {resp.status_code}: {body_preview}")

            data = resp.json()
            tool_logger.log_tool_call(f"mireye:{path}", json_body or {}, data, site_id, latency_ms, key_index)
            return data

        raise MireyeRequestFailed(
            f"Mireye {path} failed after {MAX_RETRIES} attempts"
            + (f" ({retries_429} HTTP 429)" if retries_429 else "")
            + (f" ({retries_402} HTTP 402)" if retries_402 else "")
        ) from last_error

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
        so a caller never has to know about the live API's 50-field cap.

        Chunks are issued concurrently, not sequentially: for the standard ~61-field Fire
        Intelligence Set that's only 2 chunks (50 + 11), but each chunk was a full round
        trip against the same backend that legitimately takes seconds under load (see
        DECISIONS.md), so doing them serially doubled every sample's latency for no reason -
        the chunks are independent requests to begin with. The client's own round-robin
        rate limiter is thread-safe, so this is safe at any chunk count.
        """
        chunks = _chunked(fields, MAX_FIELDS_PER_REQUEST)
        if len(chunks) == 1:
            data = self._request(
                "POST", "/v1/fetch/quote", {"lat": lat, "lng": lng, "fields": chunks[0]}, site_id=site_id
            )
            total_credits = float(data.get("credits_total", 0.0))
        else:
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                results = list(
                    pool.map(
                        lambda chunk: self._request(
                            "POST", "/v1/fetch/quote", {"lat": lat, "lng": lng, "fields": chunk}, site_id=site_id
                        ),
                        chunks,
                    )
                )
            total_credits = sum(float(d.get("credits_total", 0.0)) for d in results)
        tool_logger.log_credit_usage(site_id, quoted_credits=total_credits, actual_credits=None, key_index=None)
        return QuoteResult(credits=total_credits, fields=fields, lat=lat, lng=lng)

    def fetch(self, lat: float, lng: float, fields: list[str], site_id: str | None = None) -> dict[str, Any]:
        """Quote first (SRS FR-15 / rule 5), then fetch. Always logs the quoted cost.

        The live API charges by field-count deterministically (`credits_per_location *
        field_count`), and does not echo an "actual" cost distinct from the quote in the
        fetch response itself, so the logged actual equals the quote (SRS's "quote for
        reproducibility, not rationing" - there is no drift to reconcile here).

        Field chunks are fetched concurrently for the same reason `quote` does: they are
        independent requests, so fetching them serially only added latency.
        """
        quote_result = self.quote(lat, lng, fields, site_id=site_id)
        chunks = _chunked(fields, MAX_FIELDS_PER_REQUEST)
        merged: dict[str, Any] = {}
        if len(chunks) == 1:
            data = self._request("POST", "/v1/fetch", {"lat": lat, "lng": lng, "fields": chunks[0]}, site_id=site_id)
            merged.update(self._flatten_fields(data.get("fields", {}), data.get("fetched_at")))
        else:
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                results = list(
                    pool.map(
                        lambda chunk: self._request(
                            "POST", "/v1/fetch", {"lat": lat, "lng": lng, "fields": chunk}, site_id=site_id
                        ),
                        chunks,
                    )
                )
            for data in results:
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
