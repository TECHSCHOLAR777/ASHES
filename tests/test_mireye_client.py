import httpx
import pytest

from src.clients.mireye import MireyeAskBanned, MireyeClient, MireyeRequestFailed


def make_response(status_code=200, json_body=None):
    request = httpx.Request("POST", "https://api.mireye.com/v1/fetch")
    return httpx.Response(status_code, json=json_body or {}, request=request)


@pytest.fixture
def client():
    c = MireyeClient(["key0", "key1", "key2"])
    yield c
    c.close()


def test_round_robin_cycles_through_all_three_keys(client):
    indices = [client._pick_key_index() for _ in range(6)]
    assert indices == [0, 1, 2, 0, 1, 2]


def test_rate_limiter_skips_a_saturated_key(client):
    # Saturate key 0 to the cap (RATE_LIMIT_PER_KEY - safety margin requests in the window).
    from src.clients.mireye import RATE_LIMIT_PER_KEY, RATE_LIMIT_SAFETY_MARGIN

    for _ in range(RATE_LIMIT_PER_KEY - RATE_LIMIT_SAFETY_MARGIN):
        client._windows[0].timestamps.append(__import__("time").monotonic())

    # The round-robin counter starts back at 0 on a fresh client, so the next pick
    # should skip key 0 (saturated) and land on key 1.
    idx = client._pick_key_index()
    assert idx == 1


def test_ask_endpoint_is_banned(client):
    with pytest.raises(MireyeAskBanned):
        client._request("POST", "/v1/ask", {"q": "how far is the fire"})


def test_retries_on_429_then_succeeds(client, mocker):
    mocker.patch("time.sleep")
    responses = [make_response(429), make_response(429), make_response(200, {"lat": 1.0, "lng": 2.0})]
    mock_request = mocker.patch.object(client._client, "request", side_effect=responses)

    data = client._request("POST", "/v1/geocode", {"address": "x"})

    assert data == {"lat": 1.0, "lng": 2.0}
    assert mock_request.call_count == 3


def test_400_is_logged_before_raising(client, mocker, tmp_path, monkeypatch):
    """A 4xx must not vanish silently: it needs to appear in the tool log (NFR-14), not
    just raise. Points tool_logger at a temp dir so this test doesn't touch real logs."""
    from src.logging_ import tool_logger

    monkeypatch.setattr(tool_logger, "_LOG_DIR", tmp_path)
    mocker.patch.object(
        client._client, "request", return_value=make_response(400, {"detail": {"error": "fields_unknown"}})
    )

    with pytest.raises(MireyeRequestFailed):
        client._request("POST", "/v1/fetch", {"fields": ["bogus_field"]})

    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1
    logged_line = log_files[0].read_text(encoding="utf-8")
    assert "fields_unknown" in logged_line


def test_retries_exhausted_raises(client, mocker):
    mocker.patch("time.sleep")
    mocker.patch.object(client._client, "request", return_value=make_response(500))

    with pytest.raises(MireyeRequestFailed):
        client._request("POST", "/v1/geocode", {"address": "x"})


def test_fetch_calls_quote_before_fetch(client, mocker):
    quote_resp = make_response(200, {"credits_total": 60.0})
    fetch_resp = make_response(
        200,
        {
            "fetched_at": "2026-08-27T00:00:00Z",
            "fields": {
                "elevation": {
                    "value": 123.4,
                    "confidence": "medium",
                    "source_url": "https://www.usgs.gov/3d-elevation-program",
                    "dataset_vintage": "3DEP 1/3 arc-second",
                }
            },
        },
    )
    mock_request = mocker.patch.object(client._client, "request", side_effect=[quote_resp, fetch_resp])

    result = client.fetch(34.0, -118.0, ["elevation"], site_id="site_001")

    assert result["elevation"] == 123.4
    assert result["elevation_confidence"] == "medium"
    assert result["elevation_source_url"] == "https://www.usgs.gov/3d-elevation-program"
    assert result["elevation_vintage"] == "3DEP 1/3 arc-second"
    called_paths = [call.args[1] for call in mock_request.call_args_list]
    assert called_paths == ["/v1/fetch/quote", "/v1/fetch"]


def test_fetch_chunks_at_50_explicit_fields(client, mocker):
    fields = [f"field_{i}" for i in range(75)]  # 75 fields -> two chunks of 50 + 25
    responses = []
    for _ in range(2):  # quote chunk 1, quote chunk 2 (both inside quote()), then fetch x2
        responses.append(make_response(200, {"credits_total": 50.0}))
    for _ in range(2):
        responses.append(make_response(200, {"fetched_at": "2026-08-27T00:00:00Z", "fields": {}}))
    mock_request = mocker.patch.object(client._client, "request", side_effect=responses)

    client.fetch(34.0, -118.0, fields, site_id="site_001")

    called_bodies = [call.kwargs["json"] for call in mock_request.call_args_list]
    quote_field_counts = [len(b["fields"]) for b in called_bodies if "fields" in b]
    assert quote_field_counts == [50, 25, 50, 25]


def test_fetch_batch_chunks_at_25_and_quotes_each_chunk(client, mocker):
    coords = [(float(i), float(-i)) for i in range(30)]
    responses = []
    for _ in range(2):  # two coordinate chunks: 25 + 5
        responses.append(make_response(200, {"credits_total": 10.0}))
        responses.append(
            make_response(
                200,
                {
                    "fetched_at": "2026-08-27T00:00:00Z",
                    "results": [
                        {"index": i, "fields": {"elevation": {"value": float(i)}}}
                        for i in range(25 if _ == 0 else 5)
                    ],
                },
            )
        )
    mock_request = mocker.patch.object(client._client, "request", side_effect=responses)

    results = client.fetch_batch(coords, ["elevation"])

    assert len(results) == 30
    assert results[0]["elevation"] == 0.0
    called_paths = [call.args[1] for call in mock_request.call_args_list]
    assert called_paths == [
        "/v1/fetch/quote",
        "/v1/fetch/batch",
        "/v1/fetch/quote",
        "/v1/fetch/batch",
    ]


def test_geocode_maps_live_response_shape(client, mocker):
    resp = make_response(
        200,
        {
            "lat": 37.422398,
            "lng": -122.084212,
            "accuracy": 1.0,
            "accuracy_type": "rooftop",
            "match_type": "building_centroid",
            "normalized_address": "1600 Amphitheatre Pkwy, Mountain View, CA 94043",
            "provider": "geocodio",
            "source": "Santa Clara (Santa Clara County)",
        },
    )
    mocker.patch.object(client._client, "request", return_value=resp)

    geo = client.geocode("1600 Amphitheatre Parkway, Mountain View, CA")

    assert geo.lat == 37.422398
    assert geo.confidence == "rooftop"
    assert geo.range_interpolation is False


def test_geocode_flags_range_interpolation_for_imprecise_match(client, mocker):
    resp = make_response(200, {"lat": 1.0, "lng": 2.0, "accuracy_type": "range_interpolation"})
    mocker.patch.object(client._client, "request", return_value=resp)

    geo = client.geocode("some vague address")

    assert geo.range_interpolation is True
