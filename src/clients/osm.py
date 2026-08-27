"""OSM Overpass client for V2 road-network fire-station routing (SRS FR-57).

Queries Overpass for `amenity=fire_station` and `highway=*` ways, builds a
NetworkX graph, and returns a deterministic truck ETA. No live-traffic API;
speeds come from `config/policy.yaml` `osm_routing` plus highway class.
Processed graph is cached per bbox so the watch loop does not re-hit Overpass
every poll.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import httpx
import networkx as nx
from pyproj import Geod

from src.logging_ import tool_logger

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "fire-copilot/2.0 (contact@example.com)"
_GEOD = Geod(ellps="WGS84")

DEFAULT_SPEED_KMH = {
    "motorway": 90.0,
    "trunk": 80.0,
    "primary": 70.0,
    "secondary": 60.0,
    "tertiary": 50.0,
    "residential": 40.0,
    "unclassified": 35.0,
    "service": 25.0,
    "track": 20.0,
}


@dataclass
class OsmFireStation:
    name: str | None
    lat: float
    lng: float
    osm_id: int


@dataclass
class OsmRoute:
    eta_minutes: float
    distance_m: float
    station: OsmFireStation
    n_edges: int
    source_url: str


def _bbox(lat: float, lng: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.0
    dlng = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    return lat - dlat, lng - dlng, lat + dlat, lng + dlng


class OSMClient:
    def __init__(self, timeout: float = 40.0, url: str = OVERPASS_URL):
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
        self._url = url
        self._graph_cache: dict[str, nx.Graph] = {}
        self._station_cache: dict[str, list[OsmFireStation]] = {}

    def close(self) -> None:
        self._client.close()

    def _overpass(self, query: str, site_id: str | None) -> dict[str, Any]:
        start = time.monotonic()
        try:
            resp = self._client.post(self._url, data={"data": query})
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            tool_logger.log_tool_call("osm:overpass", {"bytes": len(query)}, None, site_id, latency_ms, error=str(exc))
            raise
        latency_ms = (time.monotonic() - start) * 1000
        tool_logger.log_tool_call(
            "osm:overpass",
            {"bytes": len(query)},
            {"elements": len(data.get("elements", []))},
            site_id,
            latency_ms,
        )
        return data

    def get_fire_stations(self, lat: float, lng: float, radius_km: float = 15.0, site_id: str | None = None) -> list[OsmFireStation]:
        key = f"st:{round(lat, 3)}:{round(lng, 3)}:{radius_km}"
        if key in self._station_cache:
            return self._station_cache[key]
        s, w, n, e = _bbox(lat, lng, radius_km)
        query = (
            "[out:json][timeout:25];"
            f"node[\"amenity\"=\"fire_station\"]({s},{w},{n},{e});"
            "out body;"
        )
        try:
            data = self._overpass(query, site_id)
        except httpx.HTTPError:
            return []
        stations = []
        for el in data.get("elements", []):
            if el.get("type") != "node":
                continue
            stations.append(
                OsmFireStation(
                    name=(el.get("tags") or {}).get("name"),
                    lat=float(el["lat"]),
                    lng=float(el["lon"]),
                    osm_id=int(el["id"]),
                )
            )
        self._station_cache[key] = stations
        return stations

    def _road_graph(self, lat: float, lng: float, radius_km: float, site_id: str | None) -> nx.Graph:
        key = f"g:{round(lat, 3)}:{round(lng, 3)}:{radius_km}"
        if key in self._graph_cache:
            return self._graph_cache[key]
        s, w, n, e = _bbox(lat, lng, radius_km)
        query = (
            "[out:json][timeout:25];"
            f"way[\"highway\"]({s},{w},{n},{e});"
            "out geom;"
        )
        graph = nx.Graph()
        try:
            data = self._overpass(query, site_id)
        except httpx.HTTPError:
            self._graph_cache[key] = graph
            return graph
        for el in data.get("elements", []):
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            highway = (el.get("tags") or {}).get("highway") or "unclassified"
            speed = DEFAULT_SPEED_KMH.get(highway, 35.0)
            for a, b in zip(geom, geom[1:]):
                n1 = (round(a["lat"], 6), round(a["lon"], 6))
                n2 = (round(b["lat"], 6), round(b["lon"], 6))
                _, _, dist_m = _GEOD.inv(n1[1], n1[0], n2[1], n2[0])
                if dist_m <= 0:
                    continue
                hours = (dist_m / 1000.0) / speed
                graph.add_edge(n1, n2, distance_m=dist_m, hours=hours, highway=highway)
        self._graph_cache[key] = graph
        return graph

    def _nearest_node(self, graph: nx.Graph, lat: float, lng: float) -> tuple[float, float] | None:
        best = None
        best_d = float("inf")
        for nlat, nlng in graph.nodes:
            _, _, d = _GEOD.inv(lng, lat, nlng, nlat)
            if d < best_d:
                best_d = d
                best = (nlat, nlng)
        return best

    def route_to_nearest_station(
        self, lat: float, lng: float, radius_km: float = 15.0, site_id: str | None = None
    ) -> OsmRoute | None:
        stations = self.get_fire_stations(lat, lng, radius_km=radius_km, site_id=site_id)
        if not stations:
            return None
        graph = self._road_graph(lat, lng, radius_km, site_id)
        if graph.number_of_edges() == 0:
            return None
        dest = self._nearest_node(graph, lat, lng)
        if dest is None:
            return None

        best: OsmRoute | None = None
        for station in stations:
            origin = self._nearest_node(graph, station.lat, station.lng)
            if origin is None:
                continue
            try:
                path = nx.shortest_path(graph, origin, dest, weight="hours")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            hours = 0.0
            dist_m = 0.0
            for u, v in zip(path, path[1:]):
                data = graph.edges[u, v]
                hours += data["hours"]
                dist_m += data["distance_m"]
            eta = hours * 60.0
            if best is None or eta < best.eta_minutes:
                best = OsmRoute(
                    eta_minutes=eta,
                    distance_m=dist_m,
                    station=station,
                    n_edges=max(0, len(path) - 1),
                    source_url="https://overpass-api.de/api/interpreter",
                )
        return best
