"""NLP layer: turn a town name / free-text ask into a coordinate with an honest mention."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from src.agents.main_agent import Site


_PLACEISH = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:,\s*[A-Z]{2})?)\b"
)


def extract_place_heuristic(question: str) -> str | None:
    """Fallback when OpenAI is off: Title-Case run, dropping leading question words."""
    skip = {
        "Is", "The", "If", "What", "When", "Where", "This", "Simulate", "Ignition",
        "Ask", "About", "Near", "Forest", "Town", "Community", "Does", "Will",
        "Can", "Could", "Should", "Would", "How",
    }
    for match in _PLACEISH.finditer(question or ""):
        parts = [p for p in match.group(1).strip().split() if p not in skip]
        if not parts:
            continue
        token = " ".join(parts)
        if len(token) < 3:
            continue
        return token
    return None


def extract_place_llm(question: str) -> dict[str, Any]:
    """Ask the copy model to name the place. Never invents lat/lng."""
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key or not (question or "").strip():
        place = extract_place_heuristic(question)
        return {
            "place": place,
            "want_simulate": "simulat" in (question or "").lower() or "ignit" in (question or "").lower(),
            "source": "heuristic",
        }
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the US place the user is asking about. Return JSON only: "
                    '{"place": string or null, "want_simulate": boolean}. '
                    "place is a geocodable town/forest/address, not a lat/lng. "
                    "Do not invent coordinates."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=120,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    place = data.get("place") if isinstance(data, dict) else None
    if not place:
        place = extract_place_heuristic(question)
    return {
        "place": place,
        "want_simulate": bool(data.get("want_simulate")) if isinstance(data, dict) else False,
        "source": "llm",
    }


def honesty_line(parsed: dict[str, Any]) -> str:
    if parsed.get("used_supplied_coords"):
        extra = f" The question also names {parsed['place']}." if parsed.get("place") else ""
        return (
            f"Using the coordinates you supplied ({parsed['lat']:.4f}, {parsed['lng']:.4f})."
            f"{extra} That is a point, not a town boundary."
        )
    if parsed.get("geocoded"):
        acc = parsed.get("confidence") or "n/a"
        return (
            f"No coordinates were supplied. Parsed “{parsed.get('place')}” from the question "
            f"and geocoded via Mireye to {parsed['lat']:.4f}, {parsed['lng']:.4f} "
            f"(accuracy {acc}). That is a point, not a town or forest boundary."
        )
    if parsed.get("error"):
        return f"Could not resolve a place: {parsed['error']}"
    return "Using the site pin already on the session."


def parse_place(deps, site: Site, question: str, coords_supplied: bool) -> dict[str, Any]:
    extracted = extract_place_llm(question)
    out: dict[str, Any] = {
        "place": extracted.get("place"),
        "want_simulate": bool(extracted.get("want_simulate")),
        "extract_source": extracted.get("source"),
        "lat": site.lat,
        "lng": site.lng,
        "used_supplied_coords": coords_supplied,
        "geocoded": False,
        "confidence": getattr(site, "geocode_confidence", None) or "n/a",
    }
    if coords_supplied:
        out["honesty"] = honesty_line(out)
        return out
    place = extracted.get("place")
    if not place:
        out["error"] = "no_place_in_question"
        out["honesty"] = honesty_line(out)
        return out
    geo = deps.mireye.geocode(place)
    site.lat, site.lng = geo.lat, geo.lng
    out.update(
        {
            "lat": geo.lat,
            "lng": geo.lng,
            "geocoded": True,
            "confidence": geo.confidence,
            "range_interpolation": geo.range_interpolation,
        }
    )
    out["honesty"] = honesty_line(out)
    return out
