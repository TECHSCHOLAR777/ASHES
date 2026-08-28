"""Sample Job C sites from a Daily progression tape.

Points come from the *growth annulus* (inside the last Daily ring, outside the
seed) plus genuine outsides of the last ring. The last ring bounds geometry; it
is never the label. ``label_point`` assigns y_72 from the tape.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from src.clients.timed_perimeters import TimedFireSeries
from src.model.arrival_labels import label_point, point_in_snapshot

SampleRole = Literal["annulus", "hard_neg", "far_neg"]


@dataclass
class SampledSite:
    lat: float
    lng: float
    role: SampleRole
    site_id: str


def _polygon_from_rings(rings: list[list[tuple[float, float]]]):
    geoms = []
    for ring in rings:
        coords = [(float(pt[0]), float(pt[1])) for pt in ring if len(pt) >= 2]
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            geoms.append(make_valid(Polygon(coords)))
        except Exception:
            continue
    if not geoms:
        return None
    return unary_union(geoms)


def _rejection_sample(
    geom,
    n: int,
    rng: random.Random,
    *,
    exclude=None,
    require_outside=None,
    max_attempts: int = 8000,
) -> list[tuple[float, float]]:
    if geom is None or geom.is_empty or n <= 0:
        return []
    minx, miny, maxx, maxy = geom.bounds
    if maxx <= minx or maxy <= miny:
        return []
    out: list[tuple[float, float]] = []
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        lng = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        pt = Point(lng, lat)
        if not geom.covers(pt):
            continue
        if exclude is not None and not exclude.is_empty and exclude.covers(pt):
            continue
        if require_outside is not None and not require_outside.is_empty and require_outside.covers(pt):
            continue
        out.append((lat, lng))
    return out


def _expand_bounds(geom, km: float, lat0: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    dlat = km / 111.0
    dlng = km / (111.0 * max(math.cos(math.radians(lat0)), 0.1))
    return minx - dlng, miny - dlat, maxx + dlng, maxy + dlat


def _box_polygon(bounds: tuple[float, float, float, float]):
    minx, miny, maxx, maxy = bounds
    return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])


def points_per_fire(n_times: int, max_acres: float, *, lo: int = 20, hi: int = 48) -> tuple[int, int, int]:
    """More timestamps / bigger fire → more sites, still capped for Mireye spend."""
    size = lo
    if n_times >= 8:
        size += 6
    if n_times >= 16:
        size += 6
    if max_acres >= 5000:
        size += 4
    if max_acres >= 20000:
        size += 4
    size = max(lo, min(hi, size))
    annulus = max(6, int(round(size * 0.40)))
    hard = max(6, int(round(size * 0.40)))
    far = max(4, size - annulus - hard)
    return annulus, hard, far


def sample_series(
    series: TimedFireSeries,
    rng: random.Random,
    n_annulus: int | None = None,
    n_hard: int | None = None,
    n_far: int | None = None,
) -> list[SampledSite]:
    snaps = sorted(series.snapshots, key=lambda s: s.t)
    if len(snaps) < 2:
        return []
    seed, last = snaps[0], snaps[-1]
    seed_geom = _polygon_from_rings(seed.geometry_rings)
    last_geom = _polygon_from_rings(last.geometry_rings)
    if last_geom is None or last_geom.is_empty:
        return []
    lat0 = (last_geom.bounds[1] + last_geom.bounds[3]) / 2.0
    max_acres = max((s.acres or 0.0) for s in snaps)
    if n_annulus is None or n_hard is None or n_far is None:
        n_annulus, n_hard, n_far = points_per_fire(len({s.t for s in snaps}), max_acres)

    annulus_pts = _rejection_sample(last_geom, n_annulus, rng, exclude=seed_geom)
    hard_box = _box_polygon(_expand_bounds(last_geom, 12.0, lat0))
    hard_pts = _rejection_sample(hard_box, n_hard, rng, require_outside=last_geom)
    far_box = _box_polygon(_expand_bounds(last_geom, 30.0, lat0))
    hard_outer = _box_polygon(_expand_bounds(last_geom, 12.0, lat0))
    far_pts = _rejection_sample(far_box, n_far, rng, require_outside=hard_outer)

    sites: list[SampledSite] = []
    for i, (lat, lng) in enumerate(annulus_pts):
        sites.append(SampledSite(lat, lng, "annulus", f"{series.fire_id}:annulus:{i}"))
    for i, (lat, lng) in enumerate(hard_pts):
        sites.append(SampledSite(lat, lng, "hard_neg", f"{series.fire_id}:hard:{i}"))
    for i, (lat, lng) in enumerate(far_pts):
        sites.append(SampledSite(lat, lng, "far_neg", f"{series.fire_id}:far:{i}"))
    # Drop anything already inside the seed (shouldn't happen; belt and suspenders).
    kept = []
    for site in sites:
        if point_in_snapshot(site.lat, site.lng, seed):
            continue
        kept.append(site)
    return kept


def early_tape_ok(series: TimedFireSeries, hours: float = 96.0) -> bool:
    """Need a second Daily ring inside `hours` of seed or 72 h labels are interval-censored junk."""
    snaps = sorted(series.snapshots, key=lambda s: s.t)
    if len(snaps) < 2:
        return False
    t0 = snaps[0].t
    n = sum(1 for s in snaps if (s.t - t0).total_seconds() / 3600.0 <= hours + 1e-6)
    return n >= 2


def labeled_rows(series: TimedFireSeries, sites: list[SampledSite]) -> list[dict]:
    rows = []
    for site in sites:
        lab = label_point(site.lat, site.lng, series)
        rows.append(
            {
                "site_id": site.site_id,
                "event_id": series.fire_id,
                "fire_id": series.fire_id,
                "fire_name": series.name,
                "site_lat": site.lat,
                "site_lng": site.lng,
                "sample_role": site.role,
                "source": series.source,
                "arrival": {
                    "n_snapshots": lab.n_snapshots,
                    "n_times": lab.n_times,
                    "source": lab.source,
                    "timed_fire_id": lab.fire_id,
                    "seed_time": lab.seed_time,
                    "last_time": lab.last_time,
                    "already_burned_at_seed": lab.already_burned_at_seed,
                    "arrival_hours": lab.arrival_hours,
                    "hit_time": lab.hit_time,
                    "series_hours": lab.series_hours,
                    "y_24": lab.y_24,
                    "y_48": lab.y_48,
                    "y_72": lab.y_72,
                    "evaluable_72": lab.evaluable_72,
                    "map_methods": lab.map_methods,
                },
            }
        )
    return rows
