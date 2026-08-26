import httpx

from src.clients.usgs import USGSClient


def make_response(json_body, status_code=200):
    request = httpx.Request("GET", "https://waterservices.usgs.gov/nwis/iv/")
    return httpx.Response(status_code, json=json_body, request=request)


def test_no_gage_id_returns_unknown():
    client = USGSClient()
    result = client.get_gage_discharge(None)
    assert result.discharge_class == "unknown"
    assert result.discharge_cfs is None
    client.close()


def test_critically_low_discharge_classified(mocker):
    client = USGSClient()
    body = {
        "value": {
            "timeSeries": [{"values": [{"value": [{"value": "5.0"}]}]}]
        }
    }
    mocker.patch.object(client._client, "get", return_value=make_response(body))

    result = client.get_gage_discharge("01234567")

    assert result.discharge_cfs == 5.0
    assert result.discharge_class == "critically_low"
    client.close()


def test_high_discharge_classified(mocker):
    client = USGSClient()
    body = {"value": {"timeSeries": [{"values": [{"value": [{"value": "1000.0"}]}]}]}}
    mocker.patch.object(client._client, "get", return_value=make_response(body))

    result = client.get_gage_discharge("01234567")

    assert result.discharge_class == "high"
    client.close()


def test_http_failure_returns_unknown(mocker):
    client = USGSClient()
    mocker.patch.object(client._client, "get", side_effect=httpx.ConnectError("down"))

    result = client.get_gage_discharge("01234567")

    assert result.discharge_class == "unknown"
    assert result.discharge_cfs is None
    client.close()


def test_empty_timeseries_returns_unknown(mocker):
    client = USGSClient()
    mocker.patch.object(client._client, "get", return_value=make_response({"value": {"timeSeries": []}}))

    result = client.get_gage_discharge("01234567")

    assert result.discharge_class == "unknown"
    client.close()
