from src.features.ros_ellipse import FAR_SENTINEL_M, compute_ros_ellipse_feature


def test_no_active_fire_returns_far_sentinel():
    result = compute_ros_ellipse_feature(
        site_lat=34.0, site_lng=-118.0, fire_lat=None, fire_lng=None,
        wind_u=1.0, wind_v=1.0, acres=100.0, dist_perim_m=5000.0,
    )
    assert result == FAR_SENTINEL_M


def test_missing_wind_falls_back_to_perimeter_distance():
    result = compute_ros_ellipse_feature(
        site_lat=34.0, site_lng=-118.0, fire_lat=34.01, fire_lng=-118.0,
        wind_u=None, wind_v=None, acres=100.0, dist_perim_m=1234.0,
    )
    assert result == 1234.0


def test_site_directly_downwind_close_in_is_inside_ellipse():
    # Strong wind blowing due north (u=0, v positive -> toward north) over 72h projects a
    # very long ellipse; a site 1 km due north of the fire should land inside it.
    result = compute_ros_ellipse_feature(
        site_lat=34.009, site_lng=-118.0,  # ~1 km north of the fire
        fire_lat=34.0, fire_lng=-118.0,
        wind_u=0.0, wind_v=10.0,  # 10 m/s wind toward the north
        acres=500.0, dist_perim_m=1000.0,
    )
    assert result < 0


def test_site_beyond_ellipse_reach_is_outside_and_positive():
    # 10 m/s wind over 72h gives a ~21.6 km major axis; a site 60 km south (opposite the
    # wind push direction, along the ellipse's own long axis) sits well beyond its reach.
    result = compute_ros_ellipse_feature(
        site_lat=33.46, site_lng=-118.0,  # ~60 km south of the fire
        fire_lat=34.0, fire_lng=-118.0,
        wind_u=0.0, wind_v=10.0,  # wind toward the north
        acres=500.0, dist_perim_m=60000.0,
    )
    assert result > 0


def test_never_raises_on_bad_input():
    result = compute_ros_ellipse_feature(
        site_lat=float("nan"), site_lng=-118.0, fire_lat=34.0, fire_lng=-118.0,
        wind_u=1.0, wind_v=1.0, acres=100.0, dist_perim_m=42.0,
    )
    assert isinstance(result, float)
