import httpx

from src.clients.wfigs import (
    WFIGSClient,
    WFIGSPerimeter,
    distance_to_perimeter_m,
    geodesic_distance_m,
)


def make_response(json_body, status_code=200):
    request = httpx.Request("GET", "https://services3.arcgis.com/x")
    return httpx.Response(status_code, json=json_body, request=request)


def test_geodesic_distance_known_points():
    # Los Angeles to San Francisco is approximately 559 km.
    dist_m = geodesic_distance_m(34.0522, -118.2437, 37.7749, -122.4194)
    assert 550_000 < dist_m < 570_000


def test_geodesic_distance_zero_for_same_point():
    dist_m = geodesic_distance_m(34.0, -118.0, 34.0, -118.0)
    assert dist_m < 1.0


def test_distance_to_perimeter_finds_nearest_vertex():
    # A small square perimeter around (34.01, -118.01); site sits just outside it.
    perimeter = WFIGSPerimeter(
        irwin_id="abc",
        name="test fire",
        geometry_rings=[
            [(-118.02, 34.00), (-118.00, 34.00), (-118.00, 34.02), (-118.02, 34.02), (-118.02, 34.00)]
        ],
    )
    dist = distance_to_perimeter_m(34.00, -118.03, perimeter)
    assert dist < 2000  # should be close to the nearest corner, not the fallback 999999


def test_distance_to_perimeter_fallback_when_empty():
    perimeter = WFIGSPerimeter(irwin_id="abc", name="test", geometry_rings=[])
    dist = distance_to_perimeter_m(34.0, -118.0, perimeter)
    assert dist == 999_999.0


def test_get_incidents_parses_features(mocker):
    client = WFIGSClient()
    body = {
        "features": [
            {
                "attributes": {
                    "IrwinID": "IR-1",
                    "IncidentName": "Test Fire",
                    "DailyAcres": 500.0,
                    "PercentContained": 25,
                    "FireDiscoveryDateTime": "2026-08-20T00:00:00Z",
                },
                "geometry": {"x": -118.1, "y": 34.1},
            }
        ]
    }
    mocker.patch.object(client._client, "get", return_value=make_response(body))

    incidents = client.get_incidents(34.0, -118.0)

    assert len(incidents) == 1
    assert incidents[0].irwin_id == "IR-1"
    assert incidents[0].containment_pct == 0.25
    client.close()


def test_get_perimeters_parses_rings(mocker):
    client = WFIGSClient()
    body = {
        "features": [
            {
                "attributes": {"IrwinID": "IR-1", "IncidentName": "Test Fire"},
                "geometry": {"rings": [[[-118.0, 34.0], [-118.01, 34.0], [-118.01, 34.01]]]},
            }
        ]
    }
    mocker.patch.object(client._client, "get", return_value=make_response(body))

    perimeters = client.get_perimeters(34.0, -118.0)

    assert len(perimeters) == 1
    assert perimeters[0].perimeter_unofficial is True
    assert len(perimeters[0].geometry_rings[0]) == 3
    client.close()


def test_get_historic_perimeters_uses_out_sr_4326_and_final_fields(mocker):
    client = WFIGSClient()
    body = {
        "features": [
            {
                "attributes": {
                    "IRWINID": "IR-HIST",
                    "INCIDENT": "August Complex",
                    "GIS_ACRES": 100000.0,
                    "FIRE_YEAR_INT": 2020,
                    "DATE_CUR": "20201101000000",
                    "FEATURE_CA": "Wildfire Final Fire Perimeter",
                    "UNQE_FIRE_ID": "CA-123",
                },
                "geometry": {"rings": [[[-123.2, 39.5], [-123.1, 39.5], [-123.1, 39.6]]]},
            }
        ]
    }
    captured = {}

    def fake_get(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return make_response(body)

    mocker.patch.object(client._client, "get", side_effect=fake_get)
    perims = client.get_historic_perimeters(39.5, -123.2, year=2020)
    assert captured["params"]["outSR"] == "4326"
    assert "FIRE_YEAR_INT=2020" in captured["params"]["where"]
    assert len(perims) == 1
    assert perims[0].name == "August Complex"
    assert perims[0].feature_category == "Wildfire Final Fire Perimeter"
    assert perims[0].perimeter_unofficial is False
    assert perims[0].fire_year == 2020
    client.close()


def test_wfigs_http_failure_returns_empty(mocker):
    client = WFIGSClient()
    mocker.patch.object(client._client, "get", side_effect=httpx.ConnectError("down"))

    assert client.get_incidents(34.0, -118.0) == []
    assert client.get_perimeters(34.0, -118.0) == []
    assert client.get_historic_perimeters(34.0, -118.0, year=2020) == []
    client.close()
