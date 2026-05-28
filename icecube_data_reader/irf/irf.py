"""
Classes to organise energy and angular resolution of IceCube track events
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self
import numpy as np
from scipy import stats
import astropy.units as u

from icecube_data_reader.downloader import available_datasets, data_directory, I3_14
from icecube_data_reader.event_types import EventType

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InstrumentResponseFunction(ABC):
    @classmethod
    @abstractmethod
    def load(cls):
        pass


class EnergyResolution(ABC):
    pass


class AngularResolution(ABC):
    pass


class IceTrackDR2InstrumentResponseFunction(
    InstrumentResponseFunction, EnergyResolution, AngularResolution
):

    def __init__(self, path: Path, season: EventType):
        """
        :param path: Path to smearing matrix
        :type path: Path
        :param season: Detector season
        :type season: EventType
        """

        self.data = np.loadtxt(path)
        self.season = season

        # Extract true energy bins and declination bins, fixed for all Ereco distributions
        self.log_tE_bin_edges = np.sort(
            np.array(list(set(self.data[:, 0:2].flatten())))
        )
        self.log_tE_bin_centers = (
            self.log_tE_bin_edges[:-1] + np.diff(self.log_tE_bin_edges) / 2
        )
        self.tE_bin_edges = np.power(10, self.log_tE_bin_edges) << u.GeV
        self.dec_bin_edges = (
            np.sort(np.array(list(set(self.data[:, 2:4].flatten())))) << u.deg
        )
        self.sin_dec_bin_edges = np.sin(self.dec_bin_edges.to_value(u.rad))
        self.sin_dec_bin_centers = (
            self.sin_dec_bin_edges[:-1] + np.diff(self.sin_dec_bin_edges) / 2
        )

        logging.debug(f"True energy bin edges: {self.log_tE_bin_edges}")
        logging.debug(f"Dec bin edges: {self.dec_bin_edges}")

        # Break naming convention because r and t are too close on the keyboard
        self.recoE_bin_edges = np.empty(
            (self.log_tE_bin_edges.size - 1, self.sin_dec_bin_edges.size - 1),
            dtype=np.ndarray,
        )
        # Create empty array for rv_histograms storing the energy resolution
        # for each bin of true energy and true declination
        self.recoE_hists = np.empty(
            (self.log_tE_bin_edges.size - 1, self.sin_dec_bin_edges.size - 1),
            dtype=stats.rv_histogram,
        )
        # Use lists here for possibly different numbers of bins for each ereco/psf/angerr histogram
        self.psf_hists = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]
        self.ang_err_hists = [
            [[] for _ in self.sin_dec_bin_centers] for _ in self.log_tE_bin_centers
        ]

    @u.quantity_input
    def create_ang_res_at_dec(self, dec: u.rad) -> None:
        """Create angular resolution histograms at provided declination

        :param dec: Declination
        :type dec: u.rad
        """

        dec_idx = np.digitize(dec, self.dec_bin_edges) - 1
        for c_tE in range(self.log_tE_bin_centers.size):
            if not isinstance(self.recoE_hists[c_tE, dec_idx], stats.rv_histogram):
                frac_counts, bins, data = self._create_recoE_distribution(
                    c_tE, dec_idx, return_data=True
                )
                self.recoE_hists[c_tE, dec_idx] = stats.rv_histogram(
                    (frac_counts, bins), density=False
                )
                self.recoE_bin_edges[c_tE, dec_idx] = bins
            else:
                data = None
                bins = self.recoE_bin_edges[c_tE, dec_idx]
            for c_rE in range(bins.size - 1):
                self.psf_hists[c_tE][dec_idx].append([])
                self.ang_err_hists[c_tE][dec_idx].append([])
                self.create_angular_distributions(c_tE, dec_idx, c_rE, data=data)
            pass

        # TODO needs to call create_eres_at_dec to ensure eres sampling is possible
        # due to chained histograms
        pass

    @u.quantity_input
    def create_IRF_at_dec(self, dec: u.rad) -> None:
        """Create entire chain of IRF distributions at provided declination.

        :param dec: Declination
        :type dec: u.rad
        """

        self.create_ang_res_at_dec(dec)

    @u.quantity_input
    def create_eres_at_dec(self, dec: u.rad) -> None:
        """Create energy resolution histograms at provided declination

        :param dec: Declination
        :type dec: u.rad
        """
        dec_idx = np.digitize(dec, self.dec_bin_edges) - 1
        for c_tE in range(self.log_tE_bin_edges.size - 1):
            if isinstance(self.recoE_hists[c_tE, dec_idx], stats.rv_histogram):
                continue
            self._create_recoE_distribution(c_tE, dec_idx)

    @classmethod
    def load(cls, season: EventType) -> Self:
        """Create energy resolution object for provided season

        :param season: Season
        :type season: EventType
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

        return cls(path, season)

    def _create_recoE_distribution(
        self, c_e: int, c_d: int, return_data: bool = False
    ) -> tuple[np.ndarray, ...]:
        """Creates the reconstructed energy distribution for given true
        energy and declination by marginalising over the kinematic (PSF) angle
        and angular error.

        :param c_e: Index of true energy bin
        :type c_e: int
        :param c_d: Index of true declination (conversely sin(declination)) bin
        :type c_d: int
        :param return_data: If true return the relevant entries of the smearing matrix, defaults to False
        :type return_data: bool, optional
        :return: Tuple of fractional counts per bin and bin edges, optional relevant entries
        of smearing matrix
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        ereco_idx = 4

        # Get entries at relevant true energy and declination
        data = self.data[self.data[:, 0] == self.log_tE_bin_edges[c_e]]
        reduced_data = data[data[:, 2] == self.dec_bin_edges[c_d].to_value(u.deg)]

        # Create bin edges of reco energy
        bins = np.array(
            sorted(
                list(
                    set(reduced_data[:, ereco_idx]).union(
                        set(reduced_data[:, ereco_idx + 1])
                    )
                )
            )
        )

        frac_counts = np.zeros(bins.size - 1)

        # marginalise over angular quantities
        for c_b, b in enumerate(bins[:-1]):
            indices = np.nonzero(np.isclose(b, reduced_data[:, ereco_idx]))
            frac_counts[c_b] = np.sum(reduced_data[indices, -1])

        self.recoE_hists[c_e, c_d] = stats.rv_histogram(
            (frac_counts, bins), density=False
        )
        self.recoE_bin_edges[c_e, c_d] = bins

        if return_data:
            return frac_counts, bins, reduced_data
        return frac_counts, bins

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
        :param c_d: Index of true declination (conversely sin(declination)) bin
        :type c_d: int
        :param c_rE: Index of reconstructed energy bin
        :type c_rE: int
        :param data: Relevant entries (i.e. for true energy, declination)
        of the smearing matrix, defaults to None
        :type data: None | np.ndarray, optional
        :return: Tuple of fractional counts ber bin and bin edges, optional relevant data
        of smearing matrix
        :rtype: tuple[np.ndarray, ...]
        """

        psf_idx = 6

        if data is None:
            # Get entries at relevant true energy and declination
            reduced_data = self.data[self.data[:, 0] == self.log_tE_bin_edges[c_e]]
            reduced_data = reduced_data[
                reduced_data[:, 2] == self.dec_bin_edges[c_d].to_value(u.deg)
            ]

        else:
            reduced_data = data

        # Get entries at relevant reco energy
        reduced_data = reduced_data[
            reduced_data[:, 4] == self.recoE_bin_edges[c_e, c_d][c_rE]
        ]

        # Create bin edges of kinematic angle / PSF
        bins = np.array(
            sorted(
                list(
                    set(reduced_data[:, psf_idx]).union(
                        set(reduced_data[:, psf_idx + 1])
                    )
                )
            )
        )

        frac_counts = np.zeros(bins.size - 1)
        self.psf_hists[c_e][c_d][c_rE] = []
        self.ang_err_hists[c_e][c_d][c_rE] = []
        for c_b, b in enumerate(bins[:-1]):
            indices = np.nonzero(np.isclose(b, reduced_data[:, psf_idx]))
            frac_counts[c_b] = np.sum(reduced_data[indices, -1])
            hist = stats.rv_histogram((frac_counts, bins), density=False)
            self.psf_hists[c_e][c_d][c_rE].append(hist)
            psf_reduced_data = reduced_data[reduced_data[:, psf_idx] == b]
            self.ang_err_hists[c_e][c_d][c_rE].append([])
            self._create_ang_err_distribution(c_e, c_d, c_rE, c_b, psf_reduced_data)

        if return_data:
            return frac_counts, bins, reduced_data
        return frac_counts, bins

    def _create_ang_err_distribution(
        self,
        c_e: int,
        c_d: int,
        c_rE: int,
        c_psf: int,
        reduced_data: np.ndarray,
    ) -> tuple[np.ndarray, ...]:

        ang_err_idx = 8
        # Reduced data by etrue, dec, ereco, psf
        bins = np.array(
            sorted(
                list(
                    set(reduced_data[:, ang_err_idx]).union(
                        set(reduced_data[:, ang_err_idx + 1])
                    )
                )
            )
        )

        frac_counts = np.zeros(bins.size - 1)
        for c_b, b in enumerate(bins[:-1]):
            indices = np.nonzero(np.isclose(b, reduced_data[:, ang_err_idx]))
            frac_counts[c_b] = np.sum(reduced_data[indices, -1])
            hist = stats.rv_histogram((frac_counts, bins), density=False)
            self.ang_err_hists[c_e][c_d][c_rE][c_psf].append(hist)
