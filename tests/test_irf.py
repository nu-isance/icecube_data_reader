import pytest

from icecube_data_reader.irf.irf import IceTracksDR2InstrumentResponseFunction
from icecube_data_reader.event_types import IC86
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


@pytest.fixture
def irf():
    irf = IceTracksDR2InstrumentResponseFunction.load(IC86)
    irf.create_eres(show_progress=False)
    return irf


def test_eres(irf):

    Etrue = 10**2.25 * u.GeV
    et_idx = np.digitize(Etrue, irf.tE_bin_edges) - 1

    coord = SkyCoord(ra=90 * u.deg, dec=2 * u.deg, frame="icrs")
    dec_idx = np.digitize(coord.dec.deg, irf.dec_bin_edges) - 1

    recoE = irf.sample_energy(coord, Etrue, N=100_000)
    bins = irf.recoE_bin_edges[et_idx][dec_idx]

    pdf = irf.recoE_hists[et_idx][dec_idx]

    n, _ = np.histogram(recoE, bins, density=True)
    cutoff = pdf >= 1e-2

    assert np.all(pytest.approx(pdf[cutoff], abs=1e-2) == n[cutoff])


def test_orthonormal(irf):
    vec = np.array([0.0, 0.0, 1.0])
    orthonormal = irf._sample_orthonormal(vec)
    assert pytest.approx(np.dot(vec, orthonormal)) == 0.0
    assert pytest.approx(np.linalg.norm(orthonormal)) == 1.0


def test_rotation(irf):
    vec = np.array([0.0, 0.0, 1.0])
    rotated = irf._rotate_around_vector(
        vec, np.array([1.0, 0.0, 0.0]), np.pi / 2 * u.rad
    )
    assert pytest.approx(np.dot(vec, rotated)) == 0.0
    assert pytest.approx(rotated[0]) == 0
    assert pytest.approx(rotated[1]) == -1.0
    assert pytest.approx(rotated[2]) == 0.0


def test_ang_res(irf):
    coord = SkyCoord(ra=90 * u.deg, dec=5 * u.deg)
    irf.create_ang_res(dec=coord.dec, show_progress=False)

    N = 100
    Etrue = 4e4 * u.GeV
    events = irf.sample(coord, Etrue, N=N)
    assert events.N == 100
    assert np.unique(events.coords.ra).size == 100
    assert np.unique(events.coords.dec).size == 100
    assert np.unique(events.coords.separation(coord)).size == 100
