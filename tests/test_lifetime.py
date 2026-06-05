from icecube_data_reader.lifetime import IceTrackDR2LifeTime
from icecube_data_reader.event_types import DR2, IC40
import numpy as np
from astropy import units as u
import pytest

lt = IceTrackDR2LifeTime()


def test_total_lt():
    total = lt.lifetime_from_season(*DR2.available_irfs)
    for c, s in enumerate(DR2.available_irfs):
        from_array = np.sum(np.diff(lt._data[s]))
        assert pytest.approx(total[s].to_value(u.d)) == from_array


def test_lt_from_mjd():
    mjd_min = 54562.3720308 - 1.0  # 1 day before IC40 start
    mjd_max = 54563.04198119 + 0.002  # 0.002 days after 2nd interval ends,
    # i.e. inbetween to next operational time window

    obs_time = lt.lifetime_from_mjd(mjd_min, mjd_max)[IC40].to_value(u.d)
    comparison = np.sum(np.diff(lt._data[IC40])[:2])
    assert pytest.approx(obs_time) == comparison


def test_mjd_from_season():
    lt = IceTrackDR2LifeTime()
    for s in DR2.available_irfs:
        mjd_min, mjd_max = lt.mjd_from_season(s)
        assert mjd_min == lt._data[s][0, 0]
        assert mjd_max == lt.data[s][-1, 1]

    with pytest.raises(ValueError) as e:
        lt.mjd_from_season("potato")
        assert "potato" in e
