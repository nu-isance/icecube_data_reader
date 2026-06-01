"""
Classes to organise energy and angular resolution of IceCube track events
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self
from itertools import pairwise
import numpy as np
from scipy import stats
import astropy.units as u
from astropy.coordinates import SkyCoord

from icecube_data_reader.downloader import available_datasets, data_directory, I3_14
from icecube_data_reader.event_types import EventType
from icecube_data_reader.utils.utils import DummyPDF

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from line_profiler import profile


class InstrumentResponseFunction(ABC):
    @classmethod
    @abstractmethod
    def load(cls):
        pass


class EnergyResolution(ABC):
    pass


class AngularResolution(ABC):
    pass


class IceTracksDR2InstrumentResponseFunction(
    InstrumentResponseFunction, EnergyResolution, AngularResolution
):

    def __init__(self, data: np.ndarray, season: EventType):
        """
        DO NOT instantiate it via init, but rather through the class method `load`

        :param path: Path to smearing matrix
        :type path: Path
        :param season: Detector season
        :type season: EventType
        """

        self.data = data
        self.season = season

        self.etrue_idx = 0
        self.dec_idx = 2
        self.ereco_idx = 4
        self.psf_idx = 6
        self.ang_err_idx = 8

    def _post_init(self):
        # Break naming convention because r and t are too close on the keyboard
        self.recoE_bin_edges = [
            [None] * self.sin_dec_bin_centers.size
        ] * self.log_tE_bin_centers.size
        # Create empty array for rv_histograms storing the energy resolution
        # for each bin of true energy and true declination
        self.recoE_hists = [
            [None] * self.sin_dec_bin_centers.size
        ] * self.log_tE_bin_centers.size
        self.recoE_sampling = [
            [None] * self.sin_dec_bin_centers.size
        ] * self.log_tE_bin_centers.size
        # Use lists here for possibly different numbers of bins for each ereco/psf/angerr histogram
        self.psf_hists = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]
        self.psf_bin_edges = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]
        self.psf_sampling = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]
        self.ang_err_hists = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]
        self.ang_err_bin_edges = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]
        self.ang_err_sampling = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]

        self.faulty = []
        for c_e in range(self.log_tE_bin_centers.size):
            for c_d, d_l in enumerate(self.dec_bin_edges[:-1]):
                idx = np.argwhere(
                    (self.data[:, 0] == self.log_tE_bin_edges[c_e])
                    * (self.data[:, 2] == d_l)
                ).squeeze()
                reduced = self.data[idx]
                if reduced[:, -1].sum() == 0.0:
                    self.faulty.append((c_e, c_d))

    @u.quantity_input
    @profile
    def create_IRF(self) -> None:
        """Create entire chain of IRF distributions at provided declination."""

        self.create_ang_res()

    @u.quantity_input
    @profile
    def create_eres(self) -> None:
        """Create energy resolution histograms"""

        for c_tE in range(self.log_tE_bin_centers.size):
            for c_d in range(self.sin_dec_bin_centers.size):
                if isinstance(
                    self.recoE_sampling[c_tE][c_d], stats.rv_histogram
                ) or isinstance(self.recoE_sampling[c_tE][c_d], DummyPDF):
                    continue
                self._create_recoE_distribution(c_tE, c_d)

    @u.quantity_input
    @profile
    def create_ang_res(self) -> None:
        """Create angular resolution histograms"""

        for c_tE in range(self.log_tE_bin_centers.size):
            for c_d in range(self.sin_dec_bin_centers.size):
                if not isinstance(self.recoE_sampling[c_tE][c_d], stats.rv_histogram):
                    frac_counts, bins, ereco_data = self._create_recoE_distribution(
                        c_tE, c_d, return_data=True
                    )
                    self.recoE_sampling[c_tE][c_d] = stats.rv_histogram(
                        (frac_counts, bins), density=False
                    )
                    self.recoE_bin_edges[c_tE][c_d] = bins
                else:
                    ereco_data = None
                    bins = self.recoE_bin_edges[c_tE][c_d]
                # bins is reco energy, each bin is mapped to a PSF distribution
                self.psf_hists[c_tE][c_d] = [None] * (bins.size - 1)
                self.psf_bin_edges[c_tE][c_d] = [None] * (bins.size - 1)
                self.psf_sampling[c_tE][c_d] = [None] * (bins.size - 1)

                self.ang_err_hists[c_tE][c_d] = [None] * (bins.size - 1)
                self.ang_err_bin_edges[c_tE][c_d] = [None] * (bins.size - 1)
                self.ang_err_sampling[c_tE][c_d] = [None] * (bins.size - 1)
                for c_rE in range(bins.size - 1):
                    self.create_angular_distributions(c_tE, c_d, c_rE, data=ereco_data)

    @classmethod
    def load(cls, season: EventType) -> Self:
        """Create energy resolution object for provided season

        :param season: Season
        :type season: EventType
        :param dec: Declination
        :type dec: u.deg
        :return: Energy resolution
        :rtype: :py:class:`IceTrackDR2EnergyResolution`
        """

        path = (
            Path(data_directory)
            / Path(available_datasets[I3_14]["dir"])
            / Path(available_datasets[I3_14]["subdir"])
            / Path("irfs")
            / Path(f"{str(season)}_smearing.csv")
        )

        data = np.loadtxt(path)
        season = season

        # Extract true energy bins and declination bins, fixed for all Ereco distributions
        log_tE_bin_edges = np.sort(np.unique(data[:, 0:2].flatten()))
        log_tE_bin_centers = log_tE_bin_edges[:-1] + np.diff(log_tE_bin_edges) / 2
        tE_bin_edges = np.power(10, log_tE_bin_edges) << u.GeV
        dec_bin_edges = np.sort(np.unique(data[:, 2:4].flatten()))

        # use log binning for angular quantities
        data[:, 6:-1] = np.log10(data[:, 6:-1])

        # dec_idx = np.digitize(dec.to_value(u.deg), dec_bin_edges) - 1
        # data = data[data[:, 2] == dec_bin_edges[dec_idx]]
        # Remove declination bc we only use one declination bin
        # mask = [True, True, False, False, True, True, True, True, True, True, True]
        # data = data[:, mask]
        irf = cls(data, season)
        irf.log_tE_bin_edges = log_tE_bin_edges
        irf.log_tE_bin_centers = log_tE_bin_centers
        irf.tE_bin_edges = tE_bin_edges
        irf.dec_bin_edges = dec_bin_edges
        irf.sin_dec_bin_edges = np.sin(np.deg2rad(dec_bin_edges))
        irf.sin_dec_bin_centers = (
            irf.sin_dec_bin_edges[:-1] + irf.sin_dec_bin_edges[1:]
        ) / 2
        # irf.dec_idx = dec_idx
        # irf.dec_min = dec_bin_edges[dec_idx] * u.deg
        # irf.dec_max = dec_bin_edges[dec_idx + 1] * u.deg

        irf._post_init()

        return irf

    @u.quantity_input
    def sample_energy(self, coord: SkyCoord, Etrue: u.GeV, seed: int = 42, N: int = 1):
        """Simulate events

        :param coord: Source coordinate,
        assumes only one coordinate per function call
        :type coord: SkyCoord
        :param Etrue: True neutrino energy
        :type Etrue: u.GeV
        :param seed: Random seed, defaults to 42
        :type seed: int, optional
        :return: _description_
        :rtype: _type_
        """

        if Etrue.shape == () and N > 1:
            Etrue = np.full(N, Etrue.to_value(u.GeV))
        else:
            Etrue = np.atleast_1d(Etrue.to_value(u.GeV))
        coord.representation_type = "spherical"
        ra = coord.ra
        dec = coord.dec
        c_d = np.digitize(dec.deg, self.dec_bin_edges) - 1

        coord.representation_type = "cartesian"
        unit_vector = np.array([coord.x, coord.y, coord.z])
        coord.representation_type = "spherical"

        log_tE = np.log10(Etrue)
        tE_idx = np.digitize(log_tE, self.log_tE_bin_edges) - 1

        recoE_out = np.zeros(Etrue.shape)

        set_e = np.unique(tE_idx)
        for idx_e in set_e:
            _index_e = np.argwhere(idx_e == tE_idx).squeeze()
            recoE = self.recoE_sampling[idx_e][c_d].rvs(
                size=_index_e.size, random_state=seed
            )
            recoE_out[_index_e] = recoE

        if recoE_out.size == 1:
            return recoE_out[0]
        return recoE_out

    @profile
    def _create_recoE_distribution(
        self,
        c_e: int,
        c_d: int,
        return_data: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Creates the reconstructed energy distribution for given true
        energy and declination by marginalising over the kinematic (PSF) angle
        and angular error.

        :param c_e: Index of true energy bin
        :type c_e: int
        :param data: Relevant entries (i.e. for true energy, declination)
        of the smearing matrix, defaults to None
        :type return_data: bool, optional
        :return: Tuple of fractional counts per bin and bin edges, optional relevant entries
        of smearing matrix
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Get entries at relevant true energy and declination
        reduced_data = self.data[
            self.data[:, self.etrue_idx] == self.log_tE_bin_edges[c_e]
        ]
        reduced_data = reduced_data[
            reduced_data[:, self.dec_idx] == self.dec_bin_edges[c_d]
        ]

        # Create bin edges of reco energy
        bins = np.sort(
            np.unique(reduced_data[:, self.ereco_idx : self.ereco_idx + 2].flatten())
        )

        frac_counts = np.zeros(bins.size - 1)

        # marginalise over angular quantities
        for c_b, b in enumerate(bins[:-1]):
            frac_counts[c_b] = np.sum(
                reduced_data[b == reduced_data[:, self.ereco_idx], -1]
            )

        self.recoE_sampling[c_e][c_d] = stats.rv_histogram(
            (frac_counts, bins), density=False
        )
        self.recoE_bin_edges[c_e][c_d] = bins
        self.recoE_hists[c_e][c_d] = frac_counts / frac_counts.sum()

        if return_data:
            return frac_counts, bins, reduced_data
        return frac_counts, bins

    @profile
    def create_angular_distributions(
        self,
        c_e: int,
        c_d: int,
        c_rE: int,
        data: None | np.ndarray = None,
        return_data: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Creates PSF distribution for provided indices of preceeding histograms
        by marginalising over the angular error.

        :param c_e: Index of true energy bin
        :type c_e: int
        :param c_rE: Index of reconstructed energy bin
        :type c_rE: int
        :param data: Relevant entries (i.e. for true energy, declination)
        of the smearing matrix, defaults to None
        :type data: None | np.ndarray, optional
        :return: Tuple of fractional counts ber bin and bin edges, optional relevant data
        of smearing matrix
        :rtype: tuple[np.ndarray, ...]
        """

        if data is None:
            # Get entries at relevant true energy and declination
            reduced_data = self.data[self.data[:, 0] == self.log_tE_bin_edges[c_e]]
            reduced_data = reduced_data[
                reduced_data[:, self.dec_idx] == self.dec_bin_edges[c_d]
            ]

        else:
            reduced_data = data

        # Get entries at relevant reco energy
        reduced_data = reduced_data[
            reduced_data[:, self.ereco_idx] == self.recoE_bin_edges[c_e][c_d][c_rE]
        ]

        # Create bin edges of kinematic angle / PSF
        bins = np.sort(
            np.unique(reduced_data[:, self.psf_idx : self.psf_idx + 2].flatten())
        )

        if bins.size == 0:
            # Happens for Ereco bins with frac_counts = 0
            # create empty histograms for this specific chain of
            # psf and ang_err
            self.psf_bin_edges[c_e][c_d][c_rE] = np.arange(21)
            self.psf_hists[c_e][c_d][c_rE] = np.zeros(20)
            self.ang_err_sampling[c_e][c_d][c_rE] = [DummyPDF()] * 20
            self.ang_err_hists[c_e][c_d][c_rE] = [np.zeros(20)] * 20
            self.ang_err_bin_edges[c_e][c_d][c_rE] = [np.zeros(21)] * 20
            return

        frac_counts = np.zeros(bins.size - 1)
        self.psf_bin_edges[c_e][c_d][c_rE] = bins
        self.ang_err_sampling[c_e][c_d][c_rE] = [None] * (bins.size - 1)
        self.ang_err_hists[c_e][c_d][c_rE] = [None] * (bins.size - 1)
        self.ang_err_bin_edges[c_e][c_d][c_rE] = [None] * (bins.size - 1)
        # Calculate fractional count in each PSF bin by marginalising over ang_err
        for c_b, b in enumerate(bins[:-1]):
            frac_counts[c_b] = np.sum(
                reduced_data[b == reduced_data[:, self.psf_idx], -1]
            )
            psf_reduced_data = reduced_data[reduced_data[:, self.psf_idx] == b]
            self._create_ang_err_distribution(c_e, c_d, c_rE, c_b, psf_reduced_data)

        # hist = stats.rv_histogram((frac_counts, bins), density=False)
        # self.psf_sampling[c_e][c_d][c_rE] = hist
        self.psf_hists[c_e][c_d][c_rE] = frac_counts / frac_counts.sum()
        if return_data:
            return frac_counts, bins, reduced_data
        return frac_counts, bins

    @profile
    def _create_ang_err_distribution(
        self,
        c_e: int,
        c_d: int,
        c_rE: int,
        c_psf: int,
        reduced_data: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Create angular error distribution for provided true energy,
        reco energy and kinematic angle indices.

        :param c_e: Index of true energy
        :type c_e: int
        :param c_rE: Index of reconstructed energy
        :type c_rE: int
        :param c_psf: Index of kinematic angle/PSF
        :type c_psf: int
        :param reduced_data: Relevant entries of the smearing matrix
        :type reduced_data: np.ndarray
        :return: _description_
        :rtype: tuple[np.ndarray, ...]
        """

        # Reduced data by etrue, dec, ereco, psf
        bins = np.sort(
            np.unique(
                reduced_data[:, self.ang_err_idx : self.ang_err_idx + 2].flatten()
            )
        )

        self.ang_err_bin_edges[c_e][c_d][c_rE][c_psf] = bins

        frac_counts = np.zeros(bins.size - 1)
        for c_b, b in enumerate(bins[:-1]):
            frac_counts[c_b] = np.sum(
                reduced_data[b == reduced_data[:, self.ang_err_idx], -1]
            )
        # hist = stats.rv_histogram((frac_counts, bins), density=False)
        # self.ang_err_sampling[c_e][c_d][c_rE][c_psf] = hist
        self.ang_err_hists[c_e][c_d][c_rE][c_psf] = frac_counts / frac_counts.sum()


if __name__ == "__main__":
    from icecube_data_reader.event_types import IC86

    irf = IceTracksDR2InstrumentResponseFunction.load(IC86)
    irf.create_IRF()
