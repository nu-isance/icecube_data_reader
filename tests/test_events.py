from icecube_data_reader.events import IceTrackDR2Events
from icecube_data_reader.event_types import IC40
from astropy import units as u
import numpy as np
from pathlib import Path
import os

import pytest


def test_event_number():
    events = IceTrackDR2Events.from_event_files()
    assert events.N == 1643355


@pytest.fixture
def test_saving(output_directory):
    events = IceTrackDR2Events.from_event_files(IC40)
    path = events.to_file(Path(output_directory) / "ic40_events.h5")
    return (path, events)


def test_overwriting(output_directory, test_saving):
    path, events = test_saving
    time_appended = events.to_file(path)
    assert os.path.basename(time_appended) in os.listdir(output_directory)


def test_loading(test_saving):
    path, ic40 = test_saving
    loaded = IceTrackDR2Events.from_file(path)

    assert np.all(
        np.isclose(loaded.energies.to_value(u.GeV), ic40.energies.to_value(u.GeV))
    )


def test_selecting(test_saving):
    idx = 5
    _, events = test_saving
    N = events.N
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True

    e = events.energies[idx].to_value(u.GeV)
    et = events.int_types[idx]
    events.select(mask)

    assert pytest.approx(events.energies[0].to_value(u.GeV)) == e
    assert et == events.int_types[0]


def test_removing(test_saving):
    idx = 5
    _, events = test_saving
    N = events.N
    events.remove(idx)
    assert events.N == N - 1


def test_erroneous_selecting(test_saving):
    idx = 5
    events = IceTrackDR2Events.from_event_files(IC40)
    N = events.N
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True

    with pytest.raises(ValueError, match="Mask needs to be of the same length as N."):
        mask = np.zeros(N + 1, dtype=bool)
        mask[1] = 1
        events.select(mask)


def test_energy_cut(test_saving):
    _, events = test_saving

    Emin = 5e3 * u.GeV
    Emax = 7e4 * u.GeV

    events.apply_energy_cut(Emin=Emin, Emax=Emax)
    assert np.all(events.energies <= Emax)
    assert np.all(events.energies >= Emin)
