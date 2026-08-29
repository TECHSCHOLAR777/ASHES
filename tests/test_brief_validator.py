from src.schemas.action_card import ActionCard, IncidentInfo, SiteRef, WeatherInfo
from src.validator.brief_validator import safe_brief, validate_brief


def _card(**overrides) -> ActionCard:
    kwargs = dict(
        card_id="c1",
        generated_at="2026-08-27T00:00:00Z",
        site=SiteRef(site_id="s1", name="Test Site", lat=34.0, lng=-118.0),
        action="monitor",
        y_hat=0.42,
        sigma=0.2,
        baseline_y=0.1,
        incident=IncidentInfo(acres=1500.0, containment_pct=0.3, dist_perimeter_m=8000.0),
        weather=WeatherInfo(),
        model_version="v0",
        policy_version="v1.0.0",
    )
    kwargs.update(overrides)
    return ActionCard(**kwargs)


def test_brief_copying_card_numbers_is_valid():
    card = _card()
    brief = "The fire is 8000.0 meters away with 1500.0 acres burned and 0.3 containment. y_hat is 0.42."
    result = validate_brief(brief, card)
    assert result.valid is True
    assert result.offending_numbers == []


def test_brief_with_invented_number_is_invalid():
    card = _card()
    brief = "The fire is approximately 9999.0 meters away."  # not in the card
    result = validate_brief(brief, card)
    assert result.valid is False
    assert 9999.0 in result.offending_numbers


def test_brief_escalation_word_on_monitor_card_is_invalid():
    card = _card(action="monitor")
    brief = "You should evacuate immediately."
    result = validate_brief(brief, card)
    assert result.valid is False
    assert result.severity_violation is True


def test_brief_escalation_word_on_evacuate_card_is_valid():
    card = _card(action="evacuate_site")
    brief = "Evacuate the site now given the 8000.0 m distance."
    result = validate_brief(brief, card)
    assert result.severity_violation is False


def test_safe_brief_returns_fallback_on_invalid():
    card = _card()
    brief = "There are 424242.0 new invented acres burning."
    safe_text, result = safe_brief(brief, card)
    assert result.valid is False
    assert "SUPPRESSED" in safe_text


def test_safe_brief_returns_original_on_valid():
    card = _card()
    brief = "Containment is 0.3."
    safe_text, result = safe_brief(brief, card)
    assert result.valid is True
    assert safe_text == brief


def test_brief_may_copy_grounded_mireye_and_engine_numbers():
    card = _card()
    brief = "Aspect is 212.0 degrees. Engine p_burn is 0.81."
    result = validate_brief(brief, card)
    assert result.valid is False
    ok = validate_brief(
        "Aspect is 212.0 degrees. Engine p_burn is 0.81.",
        card,
        grounded={"aspects": {"B": {"aspect_degrees": 212.0}}, "engine": {"p_burn_72": 0.81}},
    )
    assert ok.valid is True


def test_brief_may_copy_rounded_forms_from_copy_these_numbers():
    card = _card()
    from src.agents.agent_tools import copyable_numbers

    grounded = {
        "aspects": {"B": {"elevation": 258.847900390625, "aspect_degrees": 1.190338134765625}},
        "copy_these_numbers": copyable_numbers({"elevation": 258.847900390625, "aspect_degrees": 1.190338134765625}),
    }
    brief = "Site elevation is 258.85 m with terrain aspect 1.19 degrees (N)."
    result = validate_brief(brief, card, grounded=grounded)
    assert result.valid is True

