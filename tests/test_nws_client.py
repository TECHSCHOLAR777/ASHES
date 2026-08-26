import httpx

from src.clients.nws import NWSClient


def make_response(status_code=200, json_body=None, text_body=None, url="https://api.weather.gov/alerts/active"):
    request = httpx.Request("GET", url)
    if text_body is not None:
        return httpx.Response(status_code, text=text_body, request=request)
    return httpx.Response(status_code, json=json_body or {}, request=request)


def test_red_flag_warning_is_flagged(mocker):
    client = NWSClient()
    feature = {
        "properties": {
            "event": "Red Flag Warning",
            "severity": "Severe",
            "urgency": "Expected",
            "description": "Critical fire weather",
            "effective": "2026-08-27T00:00:00Z",
            "expires": "2026-08-28T00:00:00Z",
        }
    }
    mocker.patch.object(client._client, "get", return_value=make_response(200, {"features": [feature]}))

    alerts = client.get_cap_alerts(34.0, -118.0)

    assert len(alerts) == 1
    assert alerts[0].is_red_flag is True
    assert alerts[0].severity == "Severe"
    client.close()


def test_non_red_flag_event_not_flagged(mocker):
    client = NWSClient()
    feature = {"properties": {"event": "Winter Weather Advisory", "severity": "Minor", "urgency": "Future"}}
    mocker.patch.object(client._client, "get", return_value=make_response(200, {"features": [feature]}))

    alerts = client.get_cap_alerts(34.0, -118.0)

    assert alerts[0].is_red_flag is False
    client.close()


def test_cap_alerts_never_dropped_on_severity(mocker):
    """Ensures the client passes severity through untouched - nothing downgrades it."""
    client = NWSClient()
    feature = {"properties": {"event": "Red Flag Warning", "severity": "Extreme", "urgency": "Immediate"}}
    mocker.patch.object(client._client, "get", return_value=make_response(200, {"features": [feature]}))

    alerts = client.get_cap_alerts(34.0, -118.0)

    assert alerts[0].severity == "Extreme"
    client.close()


def test_http_failure_returns_empty_list_not_exception(mocker):
    client = NWSClient()
    mocker.patch.object(client._client, "get", side_effect=httpx.ConnectError("down"))

    alerts = client.get_cap_alerts(34.0, -118.0)

    assert alerts == []
    client.close()


def test_spc_outlook_elevated_detection(mocker):
    client = NWSClient()
    mocker.patch.object(
        client._client, "get", return_value=make_response(200, text_body="...CRITICAL FIRE WEATHER...VALID 271200Z-")
    )

    outlook = client.get_spc_outlook()

    assert outlook is not None
    assert outlook.elevated is True
    client.close()
