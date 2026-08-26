import pytest
from pydantic import ValidationError

from src.schemas.action_card import ActionCard, IncidentInfo, SiteRef, WeatherInfo


def _base_kwargs(**overrides):
    kwargs = dict(
        card_id="c1",
        generated_at="2026-08-27T00:00:00Z",
        site=SiteRef(site_id="s1", name="Test Site", lat=34.0, lng=-118.0),
        action="monitor",
        sigma=0.2,
        baseline_y=0.1,
        incident=IncidentInfo(),
        weather=WeatherInfo(),
        model_version="v0",
        policy_version="v1.0.0",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_card_constructs():
    card = ActionCard(**_base_kwargs())
    assert card.action == "monitor"
    assert card.sigma == 0.2


def test_sigma_none_raises_validation_error():
    with pytest.raises(ValidationError):
        ActionCard(**_base_kwargs(sigma=None))


def test_invalid_action_enum_raises():
    with pytest.raises(ValidationError):
        ActionCard(**_base_kwargs(action="run_away"))


def test_acres_and_containment_default_to_none_not_fabricated():
    card = ActionCard(**_base_kwargs())
    assert card.incident.acres is None
    assert card.incident.containment_pct is None
