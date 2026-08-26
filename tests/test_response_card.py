import pytest
from pydantic import ValidationError

from src.schemas.response_card import (
    Comms,
    Evacuation,
    FireStation,
    ResponseCard,
    SiteRef,
    Structure,
    USGSGaugeSummary,
    Airport,
    WaterSource,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        card_id="rc1",
        triggered_by_action_card="c1",
        generated_at="2026-08-27T00:00:00Z",
        site=SiteRef(site_id="s1", name="Test Site", lat=34.0, lng=-118.0),
        fire_station=FireStation(name="Station 5", distance_m=3000.0, eta_minutes_estimate=3.0),
        nearest_airport=Airport(name="Test Airport", distance_m=15000.0),
        comms=Comms(),
        evacuation=Evacuation(),
        structure=Structure(),
        usgs_gage_summary=USGSGaugeSummary(),
        response_card_version="v1.0.0",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_response_card_constructs():
    card = ResponseCard(**_base_kwargs())
    assert card.response_card_version == "v1.0.0"


def test_invalid_water_source_type_raises():
    with pytest.raises(ValidationError):
        WaterSource(
            type="ocean",  # not in the closed enum
            distance_m=100.0,
            availability="high",
            source_url="https://x",
            fetched_at="2026-08-27T00:00:00Z",
        )


def test_water_sources_default_empty_list_not_fabricated():
    card = ResponseCard(**_base_kwargs())
    assert card.water_sources == []
    assert card.hazmat_sites == []
