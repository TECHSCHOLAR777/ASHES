import httpx
import networkx as nx

from src.clients.osm import OSMClient


def make_response(json_body, status_code=200):
    request = httpx.Request("POST", "https://overpass-api.de/api/interpreter")
    return httpx.Response(status_code, json=json_body, request=request)


def test_route_uses_road_graph_not_straight_line(mocker):
    client = OSMClient()
    stations = {
        "elements": [{"type": "node", "id": 1, "lat": 34.01, "lon": -118.02, "tags": {"name": "Station 1"}}]
    }
    ways = {
        "elements": [
            {
                "type": "way",
                "id": 10,
                "tags": {"highway": "primary"},
                "geometry": [
                    {"lat": 34.01, "lon": -118.02},
                    {"lat": 34.01, "lon": -118.01},
                    {"lat": 34.00, "lon": -118.01},
                ],
            }
        ]
    }

    def fake_post(url, data=None, **kwargs):
        q = (data or {}).get("data", "")
        if "fire_station" in q:
            return make_response(stations)
        return make_response(ways)

    mocker.patch.object(client._client, "post", side_effect=fake_post)
    route = client.route_to_nearest_station(34.00, -118.01, radius_km=5)
    assert route is not None
    assert route.station.name == "Station 1"
    assert route.distance_m > 0
    assert route.eta_minutes > 0
    # Path has two edges, not a geodesic shortcut.
    assert route.n_edges == 2
    client.close()


def test_no_stations_returns_none(mocker):
    client = OSMClient()
    mocker.patch.object(client._client, "post", return_value=make_response({"elements": []}))
    assert client.route_to_nearest_station(34.0, -118.0) is None
    client.close()
