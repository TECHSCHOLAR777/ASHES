from datetime import date

import httpx

from src.clients.firms import ARCHIVE_SOURCES, FIRMSClient


CSV_HEADER = "latitude,longitude,frp,acq_date,acq_time,confidence\n"


def make_text_response(text, status_code=200):
    request = httpx.Request("GET", "https://firms.modaps.eosdis.nasa.gov/api/area/csv/x")
    return httpx.Response(status_code, text=text, request=request)


def test_missing_key_returns_unavailable():
    client = FIRMSClient(map_key="")
    result = client.get_hotspots(34.0, -118.0)
    assert result.unavailable is True
    assert result.count_5km == 0
    client.close()


def test_hotspots_bucketed_by_radius(mocker):
    client = FIRMSClient(map_key="testkey")
    # One hotspot ~2km away (falls in all three radii), one ~15km away (10/20 only).
    # Only the first source (VIIRS_NOAA20_NRT) returns data; the other two sources come
    # back empty, since a real deployment rarely has all three thermal sensors detect the
    # same fire in the same pass.
    csv_text = CSV_HEADER + "34.018,-118.0,12.5,2026-08-27,0100,high\n" + "34.135,-118.0,5.0,2026-08-27,0100,nominal\n"
    mocker.patch.object(
        client._client,
        "get",
        side_effect=[make_text_response(csv_text), make_text_response(CSV_HEADER), make_text_response(CSV_HEADER)],
    )

    result = client.get_hotspots(34.0, -118.0)

    assert result.count_5km == 1
    assert result.count_10km >= 1
    assert result.frp_sum_5km == 12.5
    client.close()


def test_quota_signal_marks_unavailable(mocker):
    client = FIRMSClient(map_key="badkey")
    mocker.patch.object(client._client, "get", return_value=make_text_response("Invalid MAP_KEY, quota exceeded"))

    result = client.get_hotspots(34.0, -118.0)

    assert result.unavailable is True
    client.close()


def test_historical_hotspots_uses_archive_sources_and_date_path(mocker):
    client = FIRMSClient(map_key="testkey")
    mock_get = mocker.patch.object(client._client, "get", return_value=make_text_response(CSV_HEADER))

    client.get_historical_hotspots(39.5, -122.9, on_date=date(2020, 9, 10))

    called_urls = [call.args[0] for call in mock_get.call_args_list]
    assert len(called_urls) == len(ARCHIVE_SOURCES)
    assert all("2020-09-10" in url for url in called_urls)
    assert all(any(src in url for src in ARCHIVE_SOURCES) for url in called_urls)
    client.close()


def test_quota_status_hits_real_endpoint_shape(mocker):
    client = FIRMSClient(map_key="testkey")
    body = {"transaction_limit": 5000, "current_transactions": 58, "transaction_interval": "10 minutes"}
    mocker.patch.object(client._client, "get", return_value=make_text_response(""))
    mocker.patch.object(client._client.get.return_value, "json", return_value=body, create=True)
    mocker.patch.object(client._client.get.return_value, "raise_for_status", lambda: None, create=True)

    status = client.get_quota_status()

    assert status["transaction_limit"] == 5000
    client.close()


def test_throttle_sleeps_when_near_the_verified_limit(mocker):
    from src.clients import firms as firms_module

    client = FIRMSClient(map_key="testkey")
    mock_sleep = mocker.patch("time.sleep")
    # Fill the window to just under the limit so the next call must throttle.
    now = __import__("time").monotonic()
    for _ in range(firms_module.QUOTA_TRANSACTION_LIMIT - firms_module.QUOTA_SAFETY_MARGIN):
        client._call_times.append(now)

    client._throttle()

    assert mock_sleep.called
    client.close()


def test_no_hotspots_returns_zero_counts(mocker):
    client = FIRMSClient(map_key="testkey")
    mocker.patch.object(client._client, "get", return_value=make_text_response(CSV_HEADER))

    result = client.get_hotspots(34.0, -118.0)

    assert result.count_5km == 0
    assert result.count_10km == 0
    assert result.count_20km == 0
    assert result.unavailable is False
    client.close()
