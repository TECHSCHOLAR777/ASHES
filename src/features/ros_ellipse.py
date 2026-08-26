"""Wind-projected ROS-ellipse feature (SRS FR-19, Step 6).

V1's spread signal: not a physics engine, a single versioned scalar feature. Never raises;
on any error it falls back to the already-known perimeter distance so the pipeline keeps
running with a slightly less informative feature rather than crashing.

The ROS rate itself is a crude proxy (`ros_m_per_min = ros_wind_factor * wind_speed_ms`),
explicitly not Rothermel, exactly as SRS 5.5 specifies for V1 - the real physics is
delegated to ELMFIRE in V2 (SRS FR-20), never learned from scratch here.
"""
from __future__ import annotations

import math

from pyproj import Geod

_GEOD = Geod(ellps="WGS84")

FAR_SENTINEL_M = 999_999.0
DEFAULT_DURATION_HOURS = 72
DEFAULT_ROS_WIND_FACTOR = 0.5


def compute_ros_ellipse_feature(
    site_lat: float,
    site_lng: float,
    fire_lat: float | None,
    fire_lng: float | None,
    wind_u: float | None,
    wind_v: float | None,
    acres: float | None,
    dist_perim_m: float,
    duration_hours: float = DEFAULT_DURATION_HOURS,
    ros_wind_factor: float = DEFAULT_ROS_WIND_FACTOR,
) -> float:
    """Distance from the site to the 72h wind-projected ellipse boundary, in meters.

    Negative means the site is inside the projected ellipse (fire likely to have reached
    it by the projection horizon). Falls back to `dist_perim_m` whenever the geometry or
    wind inputs are unavailable, and never raises.
    """
    try:
        if fire_lat is None or fire_lng is None:
            return FAR_SENTINEL_M

        if wind_u is None or wind_v is None:
            return dist_perim_m

        wind_speed_ms = math.sqrt(wind_u**2 + wind_v**2)
        ros_m_per_min = ros_wind_factor * wind_speed_ms
        major_axis_m = ros_m_per_min * 60.0 * duration_hours

        if major_axis_m <= 0:
            return dist_perim_m

        # Minor axis: a conservative 1:3 length-to-breadth ratio typical of wind-driven
        # elliptical fire growth models (a crude proxy, not calibrated to Rothermel).
        minor_axis_m = major_axis_m / 3.0

        # Wind direction the fire is being pushed toward (met convention: (u,v) is the
        # velocity fire embers/heat travel with, i.e. away from where the wind blows from).
        wind_bearing_deg = math.degrees(math.atan2(wind_u, wind_v)) % 360

        fwd_azimuth, _, dist_to_fire_m = _GEOD.inv(site_lng, site_lat, fire_lng, fire_lat)
        # Angle between the site's bearing-from-fire and the wind's forward bearing.
        site_bearing_from_fire = (fwd_azimuth + 180.0) % 360
        theta = math.radians(site_bearing_from_fire - wind_bearing_deg)

        # Point-in-ellipse test in a rotated frame centered on the fire, semi-axes a (major,
        # along wind) and b (minor, cross-wind).
        a, b = major_axis_m, minor_axis_m
        x = dist_to_fire_m * math.cos(theta)
        y = dist_to_fire_m * math.sin(theta)
        ellipse_value = (x / a) ** 2 + (y / b) ** 2  # < 1 means inside the ellipse

        # Approximate radial distance from the site to the ellipse boundary along the
        # site-to-fire-center ray: r_boundary = 1 / sqrt(ellipse_value / dist^2) - dist.
        if ellipse_value <= 0:
            return dist_perim_m
        r_boundary_along_ray = dist_to_fire_m / math.sqrt(ellipse_value)
        return dist_to_fire_m - r_boundary_along_ray  # negative if inside

    except Exception:
        return dist_perim_m
