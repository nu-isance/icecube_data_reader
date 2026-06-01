import pytest

from icecube_data_reader.irf.irf import IceTracksDR2InstrumentResponseFunction
from icecube_data_reader.event_types import IC86
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


def test_eres():
    irf = IceTracksDR2InstrumentResponseFunction.load(IC86)
    irf.create_eres()

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
