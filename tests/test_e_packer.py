from src.clients.firms import FIRMSResult
from src.clients.hrrr import HRRRWeather
from src.clients.nws import CAPAlert
from src.clients.wfigs import WFIGSIncident
from src.features.e_packer import FAR_SENTINEL_M, pack_e


def test_no_wfigs_perimeter_defaults_to_far_sentinel():
    e = pack_e(
        firms=None, wfigs_incident=None, wfigs_perim_dist_m=None,
        cap_alerts=[], spc=None, hrrr=None, ros_ellipse_dist_m=FAR_SENTINEL_M,
    )
    assert e.dist_perim_m == FAR_SENTINEL_M
    assert e.acres == 0.0
    assert e.containment_pct == 1.0


def test_no_red_flag_alert_gives_rflag_false():
    e = pack_e(
        firms=None, wfigs_incident=None, wfigs_perim_dist_m=None,
        cap_alerts=[CAPAlert("Winter Weather Advisory", "Minor", "Future", "", None, None, False)],
        spc=None, hrrr=None, ros_ellipse_dist_m=FAR_SENTINEL_M,
    )
    assert e.rflag is False


def test_red_flag_alert_gives_rflag_true():
    e = pack_e(
        firms=None, wfigs_incident=None, wfigs_perim_dist_m=None,
        cap_alerts=[CAPAlert("Red Flag Warning", "Severe", "Expected", "", None, None, True)],
        spc=None, hrrr=None, ros_ellipse_dist_m=FAR_SENTINEL_M,
    )
    assert e.rflag is True


def test_firms_only_sets_perimeter_unofficial():
    firms = FIRMSResult([], [], [1, 2], 0, 0, 2, 0.0, unavailable=False)
    e = pack_e(
        firms=firms, wfigs_incident=None, wfigs_perim_dist_m=None,
        cap_alerts=[], spc=None, hrrr=None, ros_ellipse_dist_m=FAR_SENTINEL_M,
    )
    assert e.perimeter_unofficial is True
    assert e.firms_count_20km == 2


def test_wfigs_incident_copies_acres_and_containment_never_invented():
    incident = WFIGSIncident("IR-1", "Test Fire", acres=1234.5, containment_pct=0.5, discovery_datetime=None, lat=34.0, lng=-118.0)
    e = pack_e(
        firms=None, wfigs_incident=incident, wfigs_perim_dist_m=2000.0,
        cap_alerts=[], spc=None, hrrr=None, ros_ellipse_dist_m=1500.0,
    )
    assert e.acres == 1234.5
    assert e.containment_pct == 0.5
    assert e.dist_perim_m == 2000.0


def test_hrrr_weather_populates_wind_and_temp():
    weather = HRRRWeather(wind_u_10m=3.0, wind_v_10m=4.0, wind_speed_ms=5.0, wind_dir_cardinal="N", temp_c=20.0, rh_pct=30.0, hrrr_valid_time="2026-08-27T00:00:00Z")
    e = pack_e(
        firms=None, wfigs_incident=None, wfigs_perim_dist_m=None,
        cap_alerts=[], spc=None, hrrr=weather, ros_ellipse_dist_m=FAR_SENTINEL_M,
    )
    assert e.wind_speed_ms == 5.0
    assert e.temp_c == 20.0
    assert e.rh_pct == 30.0


def test_firms_unavailable_flag_propagates():
    firms = FIRMSResult([], [], [], 0, 0, 0, 0.0, unavailable=True)
    e = pack_e(
        firms=firms, wfigs_incident=None, wfigs_perim_dist_m=None,
        cap_alerts=[], spc=None, hrrr=None, ros_ellipse_dist_m=FAR_SENTINEL_M,
    )
    assert e.firms_unavailable is True
