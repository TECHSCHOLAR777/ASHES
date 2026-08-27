import json
from datetime import date
from types import SimpleNamespace

from src.clients.mtbs import MTBSFire
from src.model.train import load_samples


def _square_fire(event_id="CA1234512020") -> MTBSFire:
    ring = [(-123.0, 39.4), (-122.8, 39.4), (-122.8, 39.6), (-123.0, 39.6), (-123.0, 39.4)]
    return MTBSFire(
        event_id=event_id,
        incident_name="TEST FIRE",
        ignition_date=date(2020, 9, 8),
        acres=5000.0,
        centroid_lat=39.5,
        centroid_lng=-122.9,
        geometry_rings=[[ring]],
    )


def _row(site_id, event_id="CA1234512020", y=1, spread=None):
    e = [0.0] * 15
    e[6] = 800.0
    return {
        "site_id": site_id,
        "event_id": event_id,
        "t0": "2020-09-08T12:00:00+00:00",
        "w_vector": [0.1, 0.2],
        "w_mask": [1.0, 1.0],
        "e_vector": e,
        "y": y,
        "spread_vector": spread,
    }


def test_load_samples_prefers_spread_labeled_duplicate(tmp_path):
    original = tmp_path / "a.jsonl"
    enriched = tmp_path / "b.jsonl"
    original.write_text(json.dumps(_row("s1")) + "\n", encoding="utf-8")
    enriched.write_text(
        json.dumps(_row("s1", spread=[9.0, 2.0, 0.2, 0.4, 0.6])) + "\n",
        encoding="utf-8",
    )
    samples = load_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].spread_vector == [9.0, 2.0, 0.2, 0.4, 0.6]


def test_enrich_script_attaches_spread_without_mireye(tmp_path, monkeypatch):
    import scripts.enrich_spread_vectors as enrich

    fire = _square_fire()
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text(json.dumps(_row("s1", y=1)) + "\n", encoding="utf-8")

    class FakeField:
        def sample(self, lat, lng):
            assert 39.3 < lat < 39.7
            return SimpleNamespace(
                eta_hours=9.0,
                eta_sigma_hours=2.0,
                p_burn_24=0.2,
                p_burn_48=0.4,
                p_burn_72=0.6,
            )

    monkeypatch.setattr(enrich, "spread_field_for_fire", lambda fire, landfire: FakeField())

    class FakeMTBS:
        def get_fires_by_event_ids(self, ids):
            return [fire]

        def close(self):
            pass

    class FakeLF:
        def close(self):
            pass

    monkeypatch.setattr(enrich, "MTBSClient", lambda: FakeMTBS())
    monkeypatch.setattr(enrich, "LANDFIREClient", lambda: FakeLF())
    monkeypatch.setattr(
        "sys.argv",
        ["enrich_spread_vectors.py", "--input", str(inp), "--output", str(out)],
    )
    enrich.main()
    written = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert written["spread_vector"] == [9.0, 2.0, 0.2, 0.4, 0.6]
    assert written["coords_source"] == "reconstructed_centroid_circle"
    assert written["site_lat"] is not None
