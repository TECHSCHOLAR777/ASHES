"""Historic MTBS-fire spread fields for training (Path B / --with-spread).

LANDFIRE + `spread_run` only. Never calls Mireye. The V1 jsonl already paid for W/E;
this attaches the delegated arrival-field sample those rows were missing.

Leakage rule (SRS 6.1): the MTBS *final* perimeter is the gold label `y`, never the t0
front. Historic `spread_run` is ignited at the ignition centroid only. Live V2 still
seeds the operational WFIGS perimeter, which is the true t0 state.
"""
from __future__ import annotations

import logging
from typing import Any

from src.clients.landfire import LANDFIREClient
from src.clients.mtbs import MTBSFire
from src.clients.wfigs import WFIGSPerimeter
from src.geometry.aoi import aoi_bbox_for_fetch, build_incident_aoi
from src.model.training_data import ignition_datetime
from src.spread.client import SpreadField, spread_run

logger = logging.getLogger("fire_copilot.spread.historic")

# LFPS hangs or 300s-times-out on huge historic perimeters. Cap the fetch window
# around the ignition centroid so Path B jobs stay inside a workable tile. Live
# V2 ask/watch still uses the full wind-projected incident AOI.
MAX_HISTORIC_AOI_DEG = 0.35

# Scott & Burgan table ROS is quoted at ~2.2 m/s midflame. Path B does not re-fetch
# HRRR (still free, but not in the LANDFIRE+spread_run contract). A zero wind would
# collapse the ensemble (every member identical). This reference wind is documented,
# not a live observation.
HISTORIC_DEFAULT_WIND_U = 2.2
HISTORIC_DEFAULT_WIND_V = 0.0


def _clamp_bbox_to_centroid(
    bbox: tuple[float, float, float, float],
    lat0: float | None,
    lng0: float | None,
    max_deg: float = MAX_HISTORIC_AOI_DEG,
) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    if lat0 is None or lng0 is None:
        return bbox
    half = max_deg / 2.0
    west = max(west, lng0 - half)
    east = min(east, lng0 + half)
    south = max(south, lat0 - half)
    north = min(north, lat0 + half)
    if west >= east or south >= north:
        return bbox
    return west, south, east, north


def mtbs_as_perimeter(fire: MTBSFire) -> WFIGSPerimeter:
    rings: list[list[tuple[float, float]]] = []
    for polygon in fire.geometry_rings or []:
        rings.extend(polygon)
    return WFIGSPerimeter(
        irwin_id=fire.event_id,
        name=fire.incident_name or fire.event_id,
        geometry_rings=rings,
    )


def spread_field_for_fire(
    fire: MTBSFire,
    landfire: LANDFIREClient,
    weather: dict[str, Any] | None = None,
) -> SpreadField | None:
    """Fetch LANDFIRE once for this fire and run the out-of-process engine.

    Returns None on a real fetch/engine failure. Does not invent a field.
    Weather defaults match `scripts/build_training_set.py --with-spread` (wind 0)
    so Path B and Path A attach the same kind of delegated sample.
    """
    perim = mtbs_as_perimeter(fire)
    bbox = aoi_bbox_for_fetch(perim, None, None)
    if bbox is None:
        return None
    bbox = _clamp_bbox_to_centroid(bbox, fire.centroid_lat, fire.centroid_lng)
    t0 = ignition_datetime(fire)
    if weather is None:
        weather = {
            "wind_u": HISTORIC_DEFAULT_WIND_U,
            "wind_v": HISTORIC_DEFAULT_WIND_V,
            "rh_pct": None,
            "temp_c": None,
            "valid_time": t0.isoformat() if t0 else None,
        }
    try:
        logger.info(
            "historic spread_run %s bbox=%.4f,%.4f,%.4f,%.4f",
            fire.event_id, bbox[0], bbox[1], bbox[2], bbox[3],
        )
        stack = landfire.fetch_aoi(*bbox, site_id=fire.event_id)
        geom = build_incident_aoi(
            perim, stack, wind_u=weather.get("wind_u"), wind_v=weather.get("wind_v")
        )
        # Ignition centroid only. Do not seed the gold MTBS final perimeter.
        return spread_run(
            incident_id=fire.event_id,
            landfire=stack,
            aoi=geom,
            weather=weather,
            perimeter_rings=[],
            ignition_points=(
                [{"lat": fire.centroid_lat, "lng": fire.centroid_lng}]
                if fire.centroid_lat is not None and fire.centroid_lng is not None
                else []
            ),
            site_id=fire.event_id,
        )
    except Exception as exc:
        logger.warning("spread_run/LANDFIRE failed for fire %s: %s", fire.event_id, exc)
        return None


def spread_field_for_timed_seed(
    fire: MTBSFire,
    landfire: LANDFIREClient,
    seed_rings: list[list[tuple[float, float]]],
    weather: dict[str, Any],
) -> SpreadField | None:
    """R1 historic run: seed the first operational/IR polygon, drive with real weather.

    Does not seed the MTBS final scar. Centroid ignition is kept only if the seed
    polygon is empty so the engine still has a start.
    """
    if seed_rings:
        perim = WFIGSPerimeter(
            irwin_id=fire.event_id,
            name=fire.incident_name or fire.event_id,
            geometry_rings=seed_rings,
        )
    else:
        perim = mtbs_as_perimeter(fire)
    bbox = aoi_bbox_for_fetch(perim, weather.get("wind_u"), weather.get("wind_v"))
    if bbox is None:
        return None
    bbox = _clamp_bbox_to_centroid(bbox, fire.centroid_lat, fire.centroid_lng)
    try:
        logger.info(
            "historic timed spread_run %s bbox=%.4f,%.4f,%.4f,%.4f seed_rings=%d",
            fire.event_id, bbox[0], bbox[1], bbox[2], bbox[3], len(seed_rings),
        )
        stack = landfire.fetch_aoi(*bbox, site_id=fire.event_id)
        geom = build_incident_aoi(
            perim, stack, wind_u=weather.get("wind_u"), wind_v=weather.get("wind_v")
        )
        ignitions = []
        if not seed_rings and fire.centroid_lat is not None and fire.centroid_lng is not None:
            ignitions = [{"lat": fire.centroid_lat, "lng": fire.centroid_lng}]
        return spread_run(
            incident_id=fire.event_id,
            landfire=stack,
            aoi=geom,
            weather=weather,
            perimeter_rings=seed_rings,
            ignition_points=ignitions,
            site_id=fire.event_id,
        )
    except Exception as exc:
        logger.warning("timed spread_run/LANDFIRE failed for fire %s: %s", fire.event_id, exc)
        return None
