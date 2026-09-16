"""Arrival-time labels from a timed operational perimeter series.

t0 for the label is the first snapshot (the first mapped front). Points already
inside that seed are not a 72 h forecast. Points that never appear in a later
polygon are negatives only if the series lasts at least T hours (otherwise
censored at that horizon).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.clients.mtbs import point_in_multipolygon
from src.clients.timed_perimeters import TimedFireSeries, TimedSnapshot

HORIZONS_H = (24, 48, 72)


def _as_multipolygon(rings: list[list[tuple[float, float]]]) -> list[list[list[tuple[float, float]]]]:
    if not rings:
        return []
    return [rings]


def point_in_snapshot(lat: float, lng: float, snap: TimedSnapshot) -> bool:
    return point_in_multipolygon(lat, lng, _as_multipolygon(snap.geometry_rings))


@dataclass
class ArrivalLabel:
    n_snapshots: int
    n_times: int
    source: str
    fire_id: str | None
    seed_time: str | None
    last_time: str | None
    already_burned_at_seed: bool
    arrival_hours: float | None
    hit_time: str | None
    series_hours: float | None
    y_24: int | None
    y_48: int | None
    y_72: int | None
    evaluable_72: bool
    map_methods: list[str]


def label_point(lat: float, lng: float, series: TimedFireSeries | None) -> ArrivalLabel:
    empty = ArrivalLabel(
        n_snapshots=0,
        n_times=0,
        source="none",
        fire_id=None,
        seed_time=None,
        last_time=None,
        already_burned_at_seed=False,
        arrival_hours=None,
        hit_time=None,
        series_hours=None,
        y_24=None,
        y_48=None,
        y_72=None,
        evaluable_72=False,
        map_methods=[],
    )
    if series is None or not series.snapshots:
        return empty
    snaps = sorted(series.snapshots, key=lambda s: s.t)
    seed = snaps[0]
    last = snaps[-1]
    series_hours = (last.t - seed.t).total_seconds() / 3600.0
    methods = sorted({s.map_method for s in snaps if s.map_method})
    base = dict(
        n_snapshots=len(snaps),
        n_times=len({s.t for s in snaps}),
        source=series.source,
        fire_id=series.fire_id,
        seed_time=seed.t.isoformat(),
        last_time=last.t.isoformat(),
        series_hours=series_hours,
        map_methods=methods,
    )
    if len({s.t for s in snaps}) < 2:
        return ArrivalLabel(
            already_burned_at_seed=point_in_snapshot(lat, lng, seed),
            arrival_hours=None,
            hit_time=None,
            y_24=None,
            y_48=None,
            y_72=None,
            evaluable_72=False,
            **base,
        )
    if point_in_snapshot(lat, lng, seed):
        return ArrivalLabel(
            already_burned_at_seed=True,
            arrival_hours=0.0,
            hit_time=seed.t.isoformat(),
            y_24=None,
            y_48=None,
            y_72=None,
            evaluable_72=False,
            **base,
        )
    hit: TimedSnapshot | None = None
    for snap in snaps[1:]:
        if point_in_snapshot(lat, lng, snap):
            hit = snap
            break
    arrival_hours = None
    hit_time = None
    if hit is not None:
        arrival_hours = (hit.t - seed.t).total_seconds() / 3600.0
        hit_time = hit.t.isoformat()

    def y_at(horizon: int) -> int | None:
        if arrival_hours is not None:
            return int(arrival_hours <= horizon)
        # Never observed inside: negative only if the tape ran long enough.
        if series_hours + 1e-6 >= horizon:
            return 0
        return None

    y24, y48, y72 = y_at(24), y_at(48), y_at(72)
    return ArrivalLabel(
        already_burned_at_seed=False,
        arrival_hours=arrival_hours,
        hit_time=hit_time,
        y_24=y24,
        y_48=y48,
        y_72=y72,
        evaluable_72=y72 is not None,
        **base,
    )


def seed_rings(series: TimedFireSeries) -> list[list[tuple[float, float]]]:
    snaps = sorted(series.snapshots, key=lambda s: s.t)
    return snaps[0].geometry_rings if snaps else []


def last_rings(series: TimedFireSeries) -> list[list[tuple[float, float]]]:
    snaps = sorted(series.snapshots, key=lambda s: s.t)
    return snaps[-1].geometry_rings if snaps else []


def seed_time(series: TimedFireSeries) -> datetime | None:
    snaps = sorted(series.snapshots, key=lambda s: s.t)
    return snaps[0].t if snaps else None
