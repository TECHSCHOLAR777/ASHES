from src.clients.hrrr import HRRRClient, _wind_dir_cardinal


def test_wind_dir_cardinal_north():
    # Wind blowing FROM the north means u=0, v is negative (blowing toward -y/south).
    assert _wind_dir_cardinal(0.0, -5.0) == "N"


def test_wind_dir_cardinal_east():
    assert _wind_dir_cardinal(-5.0, 0.0) == "E"


class _FakeValue:
    def __init__(self, value):
        self._value = value

    @property
    def values(self):
        return self._value


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def sel(self, **kwargs):
        return _FakeValue(self._value)


class _FakeDataset(dict):
    def __getitem__(self, key):
        return _FakeVar(super().__getitem__(key))


def test_get_weather_computes_speed_and_temp(mocker):
    fake_herbie_instance = mocker.MagicMock()
    fake_herbie_instance.xarray.side_effect = [
        _FakeDataset({"u10": 3.0, "v10": 4.0}),
        _FakeDataset({"t2m": 300.0}),
        _FakeDataset({"r2": 40.0}),
    ]
    mocker.patch("herbie.Herbie", return_value=fake_herbie_instance)

    client = HRRRClient()
    weather = client.get_weather(34.0, -118.0)

    assert weather.wind_speed_ms == 5.0
    assert round(weather.temp_c, 2) == round(300.0 - 273.15, 2)
    assert weather.rh_pct == 40.0
    assert weather.stale is False


def test_get_weather_falls_back_to_previous_hour(mocker):
    fake_ok = mocker.MagicMock()
    fake_ok.xarray.side_effect = [
        _FakeDataset({"u10": 1.0, "v10": 0.0}),
        _FakeDataset({"t2m": 290.0}),
        _FakeDataset({"r2": 50.0}),
    ]

    def herbie_side_effect(*args, **kwargs):
        # First call (current hour) raises; second call (previous hour) succeeds.
        if herbie_side_effect.calls == 0:
            herbie_side_effect.calls += 1
            raise RuntimeError("grib not yet on NODD")
        return fake_ok

    herbie_side_effect.calls = 0
    mocker.patch("herbie.Herbie", side_effect=herbie_side_effect)

    client = HRRRClient()
    weather = client.get_weather(34.0, -118.0)

    assert weather.stale is True
    assert weather.wind_speed_ms == 1.0
