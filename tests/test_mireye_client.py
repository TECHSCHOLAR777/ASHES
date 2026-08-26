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
    # Saturate key 0 to the cap (59 requests within the window).
    for _ in range(59):
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


def test_retries_exhausted_raises(client, mocker):
    mocker.patch("time.sleep")
    mocker.patch.object(client._client, "request", return_value=make_response(500))

    with pytest.raises(MireyeRequestFailed):
        client._request("POST", "/v1/geocode", {"address": "x"})


def test_fetch_calls_quote_before_fetch(client, mocker):
    quote_resp = make_response(200, {"credits": 60.0})
    fetch_resp = make_response(200, {"credits_used": 60.0, "elevation": 123.4})
    mock_request = mocker.patch.object(client._client, "request", side_effect=[quote_resp, fetch_resp])

    result = client.fetch(34.0, -118.0, ["elevation"], site_id="site_001")

    assert result["elevation"] == 123.4
    called_paths = [call.args[1] for call in mock_request.call_args_list]
    assert called_paths == ["/v1/fetch/quote", "/v1/fetch"]


def test_fetch_batch_chunks_at_25_and_quotes_each_chunk(client, mocker):
    coords = [(float(i), float(-i)) for i in range(30)]
    responses = []
    for _ in range(2):  # two chunks: 25 + 5
        responses.append(make_response(200, {"credits": 10.0}))
        responses.append(make_response(200, {"credits_used": 10.0, "results": [{"ok": True}]}))
    mock_request = mocker.patch.object(client._client, "request", side_effect=responses)

    results = client.fetch_batch(coords, ["elevation"])

    assert len(results) == 2  # one result dict appended per chunk in this mock
    called_paths = [call.args[1] for call in mock_request.call_args_list]
    assert called_paths == [
        "/v1/fetch/quote",
        "/v1/fetch/batch",
        "/v1/fetch/quote",
        "/v1/fetch/batch",
    ]
