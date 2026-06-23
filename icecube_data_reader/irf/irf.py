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
from astropy.time import Time
from tqdm.notebook import tqdm
from icecube_data_reader.downloader import available_datasets, data_directory, I3_14
from icecube_data_reader.event_types import EventType
from icecube_data_reader.utils.utils import DummyPDF
from icecube_data_reader.events import IceTrackDR2Events
from itertools import product

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


class IceTracksDR2InstrumentResponseFunction(
    InstrumentResponseFunction, EnergyResolution, AngularResolution
):

    _STACK = {}

    def __init__(
        self,
        data: np.ndarray,
        season: EventType,
        random_state: np.random.Generator = np.random.default_rng(seed=42),
    ):
        """
        DO NOT instantiate it via init, but rather through the class method `load`

        :param path: Path to smearing matrix
        :type path: Path
        :param season: Detector season
        :type season: EventType
        """

        self._data = data
        self._season = season
        self._random = random_state

        self.etrue_idx = 0
        self.dec_idx = 2
        self.ereco_idx = 4
        self.psf_idx = 6
        self.ang_err_idx = 8

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, val: np.random.Generator):
        if not isinstance(val, np.random.Generator):
            raise ValueError("random must be instance of `np.random.Generator`")
        self._random = val

    @property
    def data(self):
        return self._data

    @property
    def season(self):
        return self._season

    def _post_init(self):
        # Break naming convention because r and t are too close on the keyboard
        self.recoE_bin_edges = np.zeros(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size, 21),
        )
        # Create empty array for rv_histograms storing the energy resolution
        # for each bin of true energy and true declination
        self.recoE_hists = np.zeros(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size, 20),
        )
        self.recoE_sampling = np.empty(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size),
            dtype=stats.rv_histogram,
        )
        self.ang_err_bin_edges = np.zeros(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size, 21),
        )
        self.ang_err_hists = np.zeros(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size, 20, 20, 20),
        )
        self.psf_bin_edges = np.zeros(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size, 21)
        )
        self.psf_hists = np.zeros(
            (self.log_tE_bin_centers.size, self.sin_dec_bin_centers.size, 20, 20)
        )

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
    def create_IRF(
        self,
        dec: u.Quantity[u.deg] | None = None,
        dec_range: tuple[u.Quantity[u.deg], u.Quantity[u.deg]] = (
            -90.0 * u.deg,
            90.0 * u.deg,
        ),
        show_progress: bool = True,
    ) -> None:
        """Create the entire IRF, i.e. energy and angular resolution.

        Selection by declination or declination range is possible. A single value of `dec` takes precedent
        over a provided range.

        :param dec: Declination, defaults to None
        :type dec: u.Quantity[u.deg] | None, optional
        :param dec_range: Declination range, defaults to (-90. * u.deg, 90. * u.deg)
        :type dec_range: tuple[u.Quantity[u.deg], u.Quantity[u.deg]], optional
        :param show_progress: Display progress bar, defaults to True
        :type show_progress: bool, optional
        """

        self.create_ang_res(
            dec, dec_range, show_progress, desc="Energy and angular resolution"
        )

    @u.quantity_input
    def create_eres(
        self,
        dec: u.Quantity[u.deg] | None = None,
        dec_range: tuple[u.Quantity[u.deg], u.Quantity[u.deg]] = (
            -90.0 * u.deg,
            90.0 * u.deg,
        ),
        show_progress: bool = True,
        desc: str = "Energy resolution",
    ) -> None:
        """Create the energy resolution.

        Selection by declination or declination range is possible. A single value of `dec` takes precedent
        over a provided range.

        :param dec: Declination, defaults to None
        :type dec: u.Quantity[u.deg] | None, optional
        :param dec_range: Declination range, defaults to (-90. * u.deg, 90. * u.deg)
        :type dec_range: tuple[u.Quantity[u.deg], u.Quantity[u.deg]], optional
        :param show_progress: Display progress bar, defaults to True
        :type show_progress: bool, optional
        :param desc: Description string for progress bar, defaults to "Energy resolution"
        :type desc: str, optional
        """

        if dec is not None:
            dec_idx = np.digitize(dec.to_value(u.deg), self.dec_bin_edges) - 1
            dec_range = range(dec_idx, dec_idx + 1)
            dec_size = 1
        else:
            dec_min, dec_max = dec_range
            dec_min_idx = np.digitize(dec_min.to_value(u.deg), self.dec_bin_edges) - 1
            dec_max_idx = np.digitize(dec_max.to_value(u.deg), self.dec_bin_edges) - 1
            dec_idx = (dec_min_idx, dec_max_idx)
            dec_range = range(dec_min_idx, dec_max_idx)
            dec_size = dec_max_idx - dec_min_idx

        for c_d, c_tE in tqdm(
            product(dec_range, range(self.log_tE_bin_centers.size)),
            disable=not show_progress,
            desc=desc,
            total=self.log_tE_bin_centers.size * dec_size,
        ):
            if isinstance(
                self.recoE_sampling[c_tE][c_d], stats.rv_histogram
            ) or isinstance(self.recoE_sampling[c_tE][c_d], DummyPDF):
                continue
            self._create_recoE_distribution(c_tE, c_d)

    @u.quantity_input
    def create_ang_res(
        self,
        dec: u.Quantity[u.deg] | None = None,
        dec_range: tuple[u.Quantity[u.deg], u.Quantity[u.deg]] = (
            -90.0 * u.deg,
            90.0 * u.deg,
        ),
        show_progress: bool = True,
        desc: str = "Angular resolution",
    ) -> None:
        """Create angular resolution distributions.
        The intermediate step of kinematic angle / PSF is irrelevant,
        as it is not an observable. We skip this explicit simulation step by marginalising over it.
        For each Etrue, DEC pair, the binning of ang_err is fixed. Hence we collect the
        ang_err_bin_edges and find the fractional counts only for specific values of reconstructed energy.
        If the required energy resolution has not been created, it will be done automatically.

        Selection by declination or declination range is possible. A single value of `dec` takes precedent
        over a provided range.

        :param dec: Declination, defaults to None
        :type dec: u.Quantity[u.deg] | None, optional
        :param dec_range: Declination range, defaults to (-90. * u.deg, 90. * u.deg)
        :type dec_range: tuple[u.Quantity[u.deg], u.Quantity[u.deg]], optional
        :param show_progress: Display progress bar, defaults to True
        :type show_progress: bool, optional
        :param desc: Description string for progress bar, defaults to "Angular resolution"
        :type desc: str, optional
        :raises AssertionError: If not all ang_err_bin_edges within one pair of Etrue and DEC
        are the same, an AssertionError is raised
        """

        if dec is not None:
            dec_idx = np.digitize(dec.to_value(u.deg), self.dec_bin_edges) - 1
            dec_range = range(dec_idx, dec_idx + 1)
            dec_size = 1
        else:
            dec_min, dec_max = dec_range
            dec_min_idx = np.digitize(dec_min.to_value(u.deg), self.dec_bin_edges) - 1
            dec_max_idx = np.digitize(dec_max.to_value(u.deg), self.dec_bin_edges) - 1
            dec_idx = (dec_min_idx, dec_max_idx)
            dec_range = range(dec_min_idx, dec_max_idx)
            dec_size = dec_max_idx - dec_min_idx

        for c_d, c_tE in tqdm(
            product(dec_range, range(self.log_tE_bin_centers.size)),
            disable=not show_progress,
            desc=desc,
            total=self.log_tE_bin_centers.size * dec_size,
        ):
            if isinstance(self.recoE_sampling[c_tE][c_d], stats.rv_histogram):
                etrue_data = self.data[
                    self.data[:, self.etrue_idx] == self.log_tE_bin_edges[c_tE]
                ]
                data = etrue_data[
                    etrue_data[:, self.dec_idx] == self.dec_bin_edges[c_d]
                ]
            elif isinstance(self.recoE_sampling[c_tE][c_d], DummyPDF):
                self.psf_bin_edges[c_tE, c_d] = np.arange(21)
                self.ang_err_bin_edges[c_tE, c_d] = np.arange(21)
                data = None
            else:
                _, _, data = self._create_recoE_distribution(
                    c_tE, c_d, return_data=True
                )

            all_ang_err_bins = []
            all_psf_bins = []
            if data is None:
                self.psf_bin_edges[c_tE, c_d] = np.arange(21)
                self.ang_err_bin_edges[c_tE, c_d] = np.arange(21)
                continue
            for c_rE, rE in enumerate(self.recoE_bin_edges[c_tE, c_d][:-1]):
                red_data = data[data[:, self.ereco_idx] == rE]
                all_psf_bins.append(
                    np.unique(red_data[:, self.psf_idx : self.psf_idx + 2].flatten())
                )
                psf_counts = np.zeros(all_psf_bins[-1].size - 1)
                all_ang_err_bins.append(
                    np.unique(
                        red_data[:, self.ang_err_idx : self.ang_err_idx + 2].flatten()
                    )
                )
                ang_err_counts = np.zeros(all_ang_err_bins[-1].size - 1)
                for c_p, p in enumerate(all_psf_bins[-1][:-1]):
                    psf_red_data = red_data[red_data[:, self.psf_idx] == p]
                    psf_counts[c_p] = psf_red_data[:, -1].sum()
                    for c_a, a in enumerate(all_ang_err_bins[-1][:-1]):
                        count = psf_red_data[
                            psf_red_data[:, self.ang_err_idx] == a
                        ].squeeze()
                        ang_err_counts[c_a] = count[-1]
                    ang_err_hist = ang_err_counts.copy()
                    if np.any(ang_err_hist):
                        self.ang_err_hists[c_tE, c_d, c_rE, c_p] = (
                            ang_err_hist
                            / (ang_err_hist * np.diff(all_ang_err_bins[-1])).sum()
                        )
                psf_hist = psf_counts.copy()
                if np.any(psf_hist):
                    self.psf_hists[c_tE, c_d, c_rE] = (
                        psf_hist / (psf_hist * np.diff(all_psf_bins[-1])).sum()
                    )
                # repeat for marginalised ang_err (TODO: replace properly)
                for c_ar, ar in enumerate(all_ang_err_bins[-1][:-1]):
                    ang_err_counts[c_ar] = red_data[
                        red_data[:, self.ang_err_idx] == ar, -1
                    ].sum()
            for low, high in pairwise(all_ang_err_bins):
                if not np.all(low == high):
                    # Has been tested in a notebook, should be fine!
                    raise AssertionError(
                        "Not all ang_err bins are the same! Investigate manually and fix me."
                    )
            for low, high in pairwise(all_psf_bins):
                if not np.all(low == high):
                    # Has been tested in a notebook, should be fine!
                    raise AssertionError(
                        "Not all psf bins are the same! Investigate manually and fix me."
                    )

            self.ang_err_bin_edges[c_tE, c_d] = all_ang_err_bins[0].copy()
            self.psf_bin_edges[c_tE, c_d] = all_psf_bins[0].copy()

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

        data = np.loadtxt(path)
        season = season

        # Extract true energy bins and declination bins, fixed for all Ereco distributions
        log_tE_bin_edges = np.sort(np.unique(data[:, 0:2].flatten()))
        log_tE_bin_centers = log_tE_bin_edges[:-1] + np.diff(log_tE_bin_edges) / 2
        tE_bin_edges = np.power(10, log_tE_bin_edges) << u.GeV
        dec_bin_edges = np.sort(np.unique(data[:, 2:4].flatten()))

        # use log binning for angular quantities
        data[:, 6:-1] = np.log10(data[:, 6:-1])
        irf = cls(data, season)
        if season in cls._STACK:
            logger.info(f"loading IRF for {season} from stack")
            irf.__dict__ = cls._STACK[season].__dict__
            return irf
        else:
            cls._STACK[season] = irf
        irf.log_tE_bin_edges = log_tE_bin_edges
        irf.log_tE_bin_centers = log_tE_bin_centers
        irf.tE_bin_edges = tE_bin_edges
        irf.dec_bin_edges = dec_bin_edges
        irf.sin_dec_bin_edges = np.sin(np.deg2rad(dec_bin_edges))
        irf.sin_dec_bin_centers = (
            irf.sin_dec_bin_edges[:-1] + irf.sin_dec_bin_edges[1:]
        ) / 2

        irf._post_init()

        return irf

    @u.quantity_input
    def sample_energy(
        self,
        coord: SkyCoord,
        Etrue: u.GeV,
        N: int = 1,
    ) -> np.ndarray:
        """Sample reco energy

        :param coord: Source coordinate,
        assumes only one coordinate per function call
        :type coord: SkyCoord
        :param Etrue: True neutrino energy
        :type Etrue: u.GeV
        :param N: Number of events to sample if coord and Etrue are single values, defaults to 1
        :type N: int, optional
        :return: Array of reconstructed energies
        :rtype: np.ndarray
        """

        _, _, recoE_out = self._sample_energy(coord, Etrue, N)
        return recoE_out

    @u.quantity_input
    def sample(self, coord: SkyCoord, Etrue: u.GeV, N: int = 1) -> IceTrackDR2Events:
        """Sample reco energy

        :param coord: Source coordinate,
        assumes only one coordinate per function call
        :type coord: SkyCoord
        :param Etrue: True neutrino energy
        :type Etrue: u.GeV
        :param N: Number of events to sample if coord and Etrue are single values, defaults to 1
        :type N: int, optional
        :return:
        :rtype: np.ndarray
        """

        tE_idx, c_d, recoE = self._sample_energy(coord, Etrue, N)
        set_e = np.unique(tE_idx)

        recoE = np.atleast_1d(recoE)
        ang_errs_out = np.zeros(recoE.size)
        psf = np.zeros(recoE.size)

        for idx_e in set_e:
            _index_e = np.atleast_1d(np.argwhere(idx_e == tE_idx).squeeze())
            reco_idxs = (
                np.digitize(recoE[_index_e], self.recoE_bin_edges[idx_e, c_d]) - 1
            )
            set_rE = np.unique(reco_idxs)
            for idx_rE in set_rE:
                random = stats.rv_histogram(
                    (
                        self.psf_hists[idx_e, c_d, idx_rE],
                        self.psf_bin_edges[idx_e, c_d],
                    ),
                    density=True,
                )
                _index_rE = np.atleast_1d(np.argwhere(idx_rE == reco_idxs).squeeze())
                rvs = random.rvs(size=_index_rE.size, random_state=self.random)
                psf[_index_e[_index_rE]] = np.deg2rad(np.power(10, rvs))
                psf_idxs = (
                    np.digitize(
                        psf[_index_e[_index_rE]], self.psf_bin_edges[idx_e, c_d]
                    )
                    - 1
                )
                set_psf = np.unique(psf_idxs)
                for idx_psf in set_psf:
                    _index_psf = np.atleast_1d(
                        np.argwhere(idx_psf == psf_idxs).squeeze()
                    )
                    random = stats.rv_histogram(
                        (
                            self.ang_err_hists[idx_e, c_d, idx_rE, idx_psf],
                            self.ang_err_bin_edges[idx_e, c_d],
                        ),
                        density=True,
                    )
                    rvs = random.rvs(size=_index_psf.size, random_state=self.random)
                    ang_errs_out[_index_e[_index_rE[_index_psf]]] = np.power(10, rvs)

        ang_errs_out = ang_errs_out * u.deg
        psf = psf * u.rad
        # Deflecte like we do in stan: sample rotation axis orthonormal to the event
        # angle psf determines the amount of deflection, ang_err determines the reconstructed
        # angular uncertainty

        coord.representation_type = "cartesian"
        direction = np.array([coord.x, coord.y, coord.z])
        coord.representation_type = "spherical"
        new_directions = np.zeros((3, N))
        for c, angle in enumerate(psf):
            rot_axis = self._sample_orthonormal(direction)
            new_directions[:, c] = self._rotate_around_vector(
                direction, rot_axis, angle
            )
        new_coords = SkyCoord(
            x=new_directions[0],
            y=new_directions[1],
            z=new_directions[2],
            frame="icrs",
            representation_type="cartesian",
        )
        new_coords.representation_type = "spherical"
        events = IceTrackDR2Events(
            np.power(10, recoE) * u.GeV,
            new_coords,
            np.full(N, self.season),
            ang_errs_out,
            Time(np.full(N, 99.0), format="mjd"),
        )
        return events

    def _sample_orthonormal(self, x: np.ndarray) -> np.ndarray:
        """Sample a vector that is orthonormal to the input

        :param x: Input array
        :type x: np.ndarray
        :return: Vector orthonormal to input
        :rtype: np.ndarray
        """
        v = stats.norm().rvs(size=x.shape, random_state=self.random)
        projected = x * np.dot(x, v) / np.linalg.norm(x)
        ortho = v - projected
        orthonormal = ortho / np.linalg.norm(ortho)
        return orthonormal

    @u.quantity_input
    def _rotate_around_vector(
        self, rotatee: np.ndarray, axis: np.ndarray, theta: u.rad
    ) -> np.ndarray:
        """Rotate a vector around a second vector by an angle"""

        theta = theta.to_value(u.rad)
        return (
            axis * np.dot(rotatee, axis)
            + np.cos(theta) * np.cross(np.cross(axis, rotatee), axis)
            + np.sin(theta) * np.cross(axis, rotatee)
        )

    def _sample_energy(
        self, coord: SkyCoord, Etrue: u.GeV, N: int = 1
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Sample reco energy of events

        :param coord: Source coordinate,
        assumes only one coordinate per function call
        :type coord: SkyCoord
        :param Etrue: True neutrino energy
        :type Etrue: u.GeV
        :param N: Number of events to sample if coord and Etrue are single values, defaults to 1
        :type N: int, optional
        :return: Etrue indices, declination index, array of reconstructed energies
        :rtype: tuple[np.ndarray, int, np.ndarray]
        """

        if Etrue.shape == () and N > 1:
            Etrue = np.full(N, Etrue.to_value(u.GeV))
        else:
            Etrue = np.atleast_1d(Etrue.to_value(u.GeV))
        coord.representation_type = "spherical"
        dec = coord.dec
        c_d = np.digitize(dec.deg, self.dec_bin_edges) - 1

        coord.representation_type = "cartesian"
        coord.representation_type = "spherical"

        log_tE = np.log10(Etrue)
        tE_idx = np.digitize(log_tE, self.log_tE_bin_edges) - 1

        recoE_out = np.zeros(Etrue.shape)

        set_e = np.unique(tE_idx)
        for idx_e in set_e:
            _index_e = np.atleast_1d(np.argwhere(idx_e == tE_idx).squeeze())
            recoE = self.recoE_sampling[idx_e][c_d].rvs(
                size=_index_e.size, random_state=self.random
            )
            recoE_out[_index_e] = recoE

        if recoE_out.size == 1:
            return tE_idx, c_d, recoE_out[0]
        return tE_idx, c_d, recoE_out

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
        :param c_d: Index of declination bin
        :type c_d: int
        :param return_data: Set to true if the relevant entries of the smearing matrix
        are to be returned additionally, defaults to False
        :param data: Relevant entries (i.e. for true energy, declination)
        of the smearing matrix, defaults to None
        :type return_data: bool, optional
        :return: Tuple of fractional counts per bin and bin edges, optional relevant entries
        of smearing matrix
        :rtype: tuple[np.ndarray, ...]
        """

        if (c_e, c_d) in self.faulty:
            self.recoE_bin_edges[c_e, c_d] = np.arange(21)
            self.recoE_sampling[c_e, c_d] = DummyPDF()
            if return_data:
                return None, None, None
            return None, None
        # Get entries at relevant true energy and declination
        reduced_data = self.data[
            self.data[:, self.etrue_idx] == self.log_tE_bin_edges[c_e]
        ].copy()
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

        self.recoE_sampling[c_e, c_d] = stats.rv_histogram(
            (frac_counts, bins), density=False
        )
        self.recoE_bin_edges[c_e, c_d] = bins
        # summed = frac_counts.sum()
        self.recoE_hists[c_e, c_d] = frac_counts / (frac_counts * np.diff(bins)).sum()

        if return_data:
            return frac_counts, bins, reduced_data
        return frac_counts, bins
