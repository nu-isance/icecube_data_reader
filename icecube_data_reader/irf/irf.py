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


class EnergyResolution(ABC):
    pass

    @classmethod
    @abstractmethod
    def load(cls):
        pass


class IceTrackDR2EnergyResolution(EnergyResolution):

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
        self.tE_bin_edges = np.sort(np.array(list(set(self.data[:, 0:2].flatten()))))
        self.dec_bin_edges = np.sort(np.array(list(set(self.data[:, 2:4].flatten()))))
        self.sin_dec_bin_edges = np.sin(self.dec_bin_edges)

        logging.debug(f"True energy bin edges: {self.tE_bin_edges}")
        logging.debug(f"Dec bin edges: {self.dec_bin_edges}")

        # Break naming convention because r and t are too close on the keyboard
        self.recoE_bin_edges = np.empty(
            (self.tE_bin_edges.size - 1, self.sin_dec_bin_edges.size - 1),
            dtype=np.ndarray,
        )
        # Create empty array for rv_histograms storing the energy resolution
        # for each bin of true energy and true declination
        self.recoE_hists = np.empty(
            (self.tE_bin_edges.size - 1, self.sin_dec_bin_edges.size - 1),
            dtype=stats.rv_histogram,
        )

    @u.quantity_input
    def create_eres_at_dec(self, dec: u.rad) -> None:
        """Create energy resolution histograms at provided declination

        :param dec: Declination
        :type dec: u.rad
        """
        dec_idx = np.digitize(dec.to_value(u.deg), self.dec_bin_edges) - 1
        for c_tE in range(self.tE_bin_edges.size - 1):
            frac_counts, bins = self.marginalise_over_angles(c_tE, dec_idx)
            # Set density=False because smearing matrix provides unnormalised fractional counts
            hist = stats.rv_histogram((frac_counts, bins), density=False)
            self.recoE_hists[c_tE, dec_idx] = hist
            self.recoE_bin_edges[c_tE, dec_idx] = bins

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

    def marginalise_over_angles(
        self, c_e: int, c_d: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates the reconstructed energy distribution for given true
        energy and declination by marginalising over the kinematic (PSF) angle
        and angular error.

        :param c_e: Index of true energy bin
        :type c_e: int
        :param c_d: Index of true declination (conversly sin(declination) bin
        :type c_d: int
        :return: Tuple of fractional counts per bin and bin edges
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        ereco_idx = 4

        data = self.data[self.data[:, 0] == self.tE_bin_edges[c_e]]
        reduced_data = data[data[:, 2] == self.dec_bin_edges[c_d]]

        bins = np.array(
            sorted(
                list(
                    set(reduced_data[:, ereco_idx]).union(
                        set(reduced_data[:, ereco_idx + 1])
                    )
                )
            )
        )

        frac_counts = np.zeros(bins.shape[0] - 1)

        # marginalise over angular quantities
        for c_b, b in enumerate(bins[:-1]):
            indices = np.nonzero(np.isclose(b, reduced_data[:, ereco_idx]))
            frac_counts[c_b] = np.sum(reduced_data[indices, -1])

        return frac_counts, bins
