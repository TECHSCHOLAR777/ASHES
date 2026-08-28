from types import SimpleNamespace

from src.clients.firms import FIRMSResult
from src.features.e_packer import pack_e
from src.features.w_encoder import encode_w
from src.model.h_fire import model_infer


def _e():
    return pack_e(
        firms=FIRMSResult([], [], [], 0, 0, 0, 0.0, unavailable=False),
        wfigs_incident=None,
        wfigs_perim_dist_m=10_000.0,
        cap_alerts=[],
        spc=None,
        hrrr=None,
        ros_ellipse_dist_m=10_000.0,
    )


def test_model_infer_uses_raw_p72_as_v2_baseline():
    w = encode_w({})
    spread = SimpleNamespace(
        eta_hours=8.0,
        eta_sigma_hours=2.0,
        p_burn_24=0.4,
        p_burn_48=0.6,
        p_burn_72=0.75,
        spread_field_version="rothermel_huygens_v1:n7:h72:grid5x5",
    )
    out = model_infer("s1", w, _e(), spread=spread)
    assert out.baseline_y == 0.75
    assert out.eta_hours == 8.0
    assert out.eta_sigma_hours == 2.0
    assert out.p_burn_by_T["72"] == 0.75
    assert out.spread_field_version.startswith("rothermel_huygens_v1")
    assert 0.0 <= out.y_hat <= 1.0


def test_model_infer_outside_field_omits_empty_p_burn():
    w = encode_w({})
    spread = SimpleNamespace(
        eta_hours=None,
        eta_sigma_hours=None,
        p_burn_24=None,
        p_burn_48=None,
        p_burn_72=None,
        spread_field_version="rothermel_huygens_v1:n7:h72:grid5x5",
    )
    out = model_infer("s1", w, _e(), spread=spread)
    assert out.p_burn_by_T is None
    assert out.eta_hours is None
    assert out.spread_field_version.startswith("rothermel_huygens_v1")


def test_model_infer_without_spread_keeps_v1_baseline():
    w = encode_w({})
    out = model_infer("s1", w, _e(), spread=None)
    assert out.eta_hours is None
    assert out.spread_field_version is None
    assert out.p_burn_by_T is None
    assert 0.0 <= out.y_hat <= 1.0
    # A V2 pickle is 5 dims wider; missing spread must not crash infer.
    assert out.y_hat == out.y_hat  # finite
