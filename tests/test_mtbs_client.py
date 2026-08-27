import httpx

from src.clients.mtbs import MTBSClient, point_in_multipolygon


def make_response(json_body, status_code=200):
    request = httpx.Request("GET", "https://edcintl.cr.usgs.gov/geoserver/mtbs/ows")
    return httpx.Response(status_code, json=json_body, request=request)


def test_point_inside_square_polygon_is_true():
    # A simple 1-degree square from (0,0) to (1,1) in (lng, lat) pairs.
    square = [[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]]]
    assert point_in_multipolygon(0.5, 0.5, square) is True


def test_point_outside_square_polygon_is_false():
    square = [[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]]]
    assert point_in_multipolygon(5.0, 5.0, square) is False


def test_point_inside_hole_is_excluded():
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    hole = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0), (4.0, 4.0)]
    polygon_with_hole = [[outer, hole]]
    assert point_in_multipolygon(5.0, 5.0, polygon_with_hole) is False  # inside the hole
    assert point_in_multipolygon(1.0, 1.0, polygon_with_hole) is True  # inside outer, outside hole


def test_get_fires_in_bbox_parses_real_schema(mocker):
    client = MTBSClient()
    body = {
        "features": [
            {
                "properties": {
                    "event_id": "CA1234512020",
                    "incid_name": "TEST FIRE",
                    "ig_date": "2020-08-01Z",
                    "burnbndac": 5000,
                    "burnbndlat": "39.5",
                    "burnbndlon": "-121.5",
                },
                "geometry": {"type": "MultiPolygon", "coordinates": [[[[-121.6, 39.4], [-121.4, 39.4], [-121.4, 39.6], [-121.6, 39.6]]]]},
            }
        ]
    }
    mocker.patch.object(client._client, "get", return_value=make_response(body))

    fires = client.get_fires_in_bbox(-122.0, 39.0, -121.0, 40.0)

    assert len(fires) == 1
    assert fires[0].event_id == "CA1234512020"
    assert fires[0].acres == 5000.0
    assert fires[0].ignition_date.isoformat() == "2020-08-01"
    client.close()


def test_get_fires_http_failure_returns_empty(mocker):
    client = MTBSClient()
    mocker.patch.object(client._client, "get", side_effect=httpx.ConnectError("down"))

    fires = client.get_fires_in_bbox(-122.0, 39.0, -121.0, 40.0)

    assert fires == []
    client.close()
