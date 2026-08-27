"""Rejects any LLM brief that introduces a new number or alters CAP severity (SRS FR-36, Step 13).

The LLM's only job on the write step is to copy ActionCard fields into prose (SRS 4.2). This
validator is the enforcement mechanism: every number in the brief must already exist,
verbatim (within floating-point tolerance), somewhere in the ActionCard's own JSON.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.schemas.action_card import ActionCard

NUMBER_PATTERN = re.compile(r"\b\d+\.?\d*\b")
FLOAT_TOLERANCE = 0.001
SUPPRESSED_ACTIONS = {"monitor", "no_action"}
ESCALATION_WORDS = ("evacuate", "protect")

SUPPRESSED_BRIEF_MESSAGE = (
    "[BRIEF SUPPRESSED: validator caught invented numbers. See ActionCard for all data.]"
)


@dataclass
class ValidationResult:
    valid: bool
    offending_numbers: list[float] = field(default_factory=list)
    severity_violation: bool = False


def _extract_numbers(text: str) -> list[float]:
    return [float(match) for match in NUMBER_PATTERN.findall(text)]


def validate_brief(brief: str, action_card: ActionCard) -> ValidationResult:
    card_json = action_card.model_dump_json()
    card_numbers = _extract_numbers(card_json)
    brief_numbers = _extract_numbers(brief)

    offending = [
        n for n in brief_numbers if not any(abs(n - cn) <= FLOAT_TOLERANCE for cn in card_numbers)
    ]

    severity_violation = False
    if action_card.action in SUPPRESSED_ACTIONS:
        lowered = brief.lower()
        if any(word in lowered for word in ESCALATION_WORDS):
            severity_violation = True

    valid = not offending and not severity_violation
    return ValidationResult(valid=valid, offending_numbers=offending, severity_violation=severity_violation)


def safe_brief(brief: str, action_card: ActionCard) -> tuple[str, ValidationResult]:
    """Returns (brief, result) if valid, else (fallback message, result)."""
    result = validate_brief(brief, action_card)
    if result.valid:
        return brief, result
    return SUPPRESSED_BRIEF_MESSAGE, result
