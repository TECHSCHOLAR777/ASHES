from src.policy.engine import PolicyInput, apply_policy, load_policy_config

CFG = load_policy_config()


def _pin(**overrides) -> PolicyInput:
    base = dict(
        dist_perim_m=999_999.0,
        red_flag=False,
        spc_elevated=False,
        firms_count_5km=0,
        perimeter_unofficial=False,
    )
    base.update(overrides)
    return PolicyInput(**base)


def test_no_signal_no_spc_gives_no_action():
    result = apply_policy(y_hat=0.05, sigma=0.2, pin=_pin(), config=CFG)
    assert result.action == "no_action"


def test_no_signal_but_spc_elevated_gives_monitor():
    result = apply_policy(y_hat=0.05, sigma=0.2, pin=_pin(spc_elevated=True), config=CFG)
    assert result.action == "monitor"


def test_firms_without_wfigs_gives_monitor_with_unofficial_flag():
    result = apply_policy(y_hat=0.2, sigma=0.2, pin=_pin(firms_count_5km=3), config=CFG)
    assert result.action == "monitor"
    assert "perimeter_unofficial" in result.flags


def test_firms_on_developed_land_never_prepares():
    result = apply_policy(
        y_hat=0.6, sigma=0.2, pin=_pin(firms_count_5km=5, land_use_class="Developed"), config=CFG
    )
    assert result.action == "monitor"


def test_wfigs_far_and_low_yhat_gives_monitor():
    result = apply_policy(y_hat=0.1, sigma=0.2, pin=_pin(dist_perim_m=20_000.0), config=CFG)
    assert result.action == "monitor"


def test_mid_distance_red_flag_high_ndvi_moderate_yhat_gives_prepare():
    result = apply_policy(
        y_hat=0.5,
        sigma=0.2,
        pin=_pin(dist_perim_m=6_000.0, red_flag=True, ndvi_current=0.7),
        config=CFG,
    )
    assert result.action == "prepare"


def test_close_distance_low_density_gives_protect_asset():
    result = apply_policy(
        y_hat=0.5,
        sigma=0.1,
        pin=_pin(dist_perim_m=1_000.0, housing_density_per_km2=50.0, road_access_limited=False),
        config=CFG,
    )
    assert result.action == "protect_asset"


def test_close_distance_high_density_limited_egress_gives_evacuate():
    result = apply_policy(
        y_hat=0.5,
        sigma=0.1,
        pin=_pin(dist_perim_m=1_000.0, housing_density_per_km2=2000.0, road_access_limited=True),
        config=CFG,
    )
    assert result.action == "evacuate_site"


def test_high_sigma_suppresses_evacuate_to_protect_asset():
    result = apply_policy(
        y_hat=0.5,
        sigma=0.9,
        pin=_pin(dist_perim_m=1_000.0, housing_density_per_km2=2000.0, road_access_limited=True),
        config=CFG,
    )
    assert result.action == "protect_asset"
    assert "no_ros_high_sigma" in result.flags


def test_contained_and_previously_in_play_gives_inspect_after():
    result = apply_policy(
        y_hat=0.1,
        sigma=0.2,
        pin=_pin(dist_perim_m=500.0, containment_pct=1.0, previously_in_play=True),
        config=CFG,
    )
    assert result.action == "inspect_after"


def test_eta_24_to_72_hours_gives_prepare():
    result = apply_policy(
        y_hat=0.2,
        sigma=0.1,
        pin=_pin(dist_perim_m=20_000.0, eta_hours=36.0, eta_sigma_hours=2.0),
        config=CFG,
    )
    assert result.action == "prepare"


def test_imminent_eta_low_density_gives_protect_asset():
    result = apply_policy(
        y_hat=0.2,
        sigma=0.1,
        pin=_pin(dist_perim_m=20_000.0, eta_hours=3.0, eta_sigma_hours=1.0, housing_density_per_km2=10.0),
        config=CFG,
    )
    assert result.action == "protect_asset"


def test_imminent_eta_high_density_gives_evacuate():
    result = apply_policy(
        y_hat=0.2,
        sigma=0.1,
        pin=_pin(
            dist_perim_m=20_000.0,
            eta_hours=3.0,
            eta_sigma_hours=1.0,
            housing_density_per_km2=2000.0,
            road_access_limited=True,
        ),
        config=CFG,
    )
    assert result.action == "evacuate_site"


def test_high_eta_sigma_suppresses_evacuate():
    result = apply_policy(
        y_hat=0.2,
        sigma=0.1,
        pin=_pin(
            dist_perim_m=20_000.0,
            eta_hours=3.0,
            eta_sigma_hours=20.0,
            housing_density_per_km2=2000.0,
            road_access_limited=True,
        ),
        config=CFG,
    )
    assert result.action == "protect_asset"
    assert "no_ros_high_sigma" in result.flags


def test_high_yhat_far_distance_gives_protect_asset():
    result = apply_policy(
        y_hat=0.85,
        sigma=0.1,
        pin=_pin(dist_perim_m=25_000.0, housing_density_per_km2=10.0, road_access_limited=False),
        config=CFG,
    )
    assert result.action == "protect_asset"
