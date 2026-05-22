"""
Class to organise IceCube event data
"""

from abc import ABC
import os
from os.path import join
from pathlib import Path
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

import logging

from icecube_data_reader.downloader import data_directory, I3_14, available_datasets, IceCubeData
from icecube_data_reader.event_types import IC40, IC59, IC79, IC86, suffixes
from icecube_data_reader.lifetime import LifeTime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Events(ABC):
    """
    Meta class for IceCube events
    """

    @property
    def mjd(self):
        return self._mjd

    @property
    def energy(self):
        return self._energy

    @property
    def ang_err(self):
        return self._ang_err

    @property
    def coord(self):
        return self._coord

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def type(self):
        return self._type

    @property
    def N(self):
        return self.type.size

    def apply_energy_cut(self, Emin: u.GeV, Emax: u.GeV = np.inf * u.GeV):
        pass


class IceTrackDR2Events(Events):
    """
    Organise event data of the IceTracks-DR2
    """

    mjd_ = 3
    energy_ = 4
    ang_err_ = 5
    ra_ = 6
    dec_ = 7

    @u.quantity_input
    def __init__(
        self,
        energy: u.GeV,
        coord: SkyCoord,
        type: np.ndarray,
        ang_err: u.deg,
        mjd: Time,
    ):
        self._energy = energy
        self._coord = coord
        self._type = type
        self._ang_err = ang_err
        self._mjd = mjd
        self._coord.representation_type = "cartesian"
        self._unit_vector = np.array([coord.x.value, coord.y.value, coord.z.value]).T
        self._coord.representation_type = "spherical"

    def scramble_ra(self, seed: int = 42) -> None:
        """
        Scrambles the right ascension of all events to generate pseudo background data.

        :param seed: Seed for the random generator

        :returns: None
        """

        logger.warning("Scrambling RA. To revert this operation reload the events")
        rng = np.default_rng(seed=seed)
        ra = rng.random(self.ra.size) * 2 * np.pi * u.rad
        self.ra = ra.to(u.deg)
        self.coords = SkyCoord(ra=self.ra, dec=self.dec, frame="icrs")

    def scramble_mjd(self, lifetime: LifeTime, seed: int = 42) -> None:
        pass

    def select(self, mask: npt.NDArray[np.bool_]):
        """
        Select some subset of existing events by providing a mask.
        :param mask: Array of bools with same length as event properties.

        :returns: None
        """

        if not len(mask) == self.N:
            raise ValueError("Mask needs to be of the same length as N.")

        try:
            self._idxs[self._idxs] = np.logical_and(self._idxs[self._idxs], mask)
        except AttributeError:
            pass

        self._energy = self._energy[mask]
        self._coord = self._coord[mask]
        self._unit_vector = self._unit_vector[mask]
        self._type = self._type[mask]
        self._ang_err = self._ang_err[mask]
        self._mjd = self._mjd[mask]

    @classmethod
    def from_event_files(cls, *seasons):
        """
        Load data of provided seasons.
        If none are provided, use all.

        :param seasons: Seasons to load

        :returns: Event container :py:class:`icecube_data_reader.events.Events`
        """

        if seasons == ():
            seasons = (IC40, IC59, IC79, IC86)

        logger.debug(f"Loading seasons: {[_ for _ in seasons]}")

        directory = available_datasets[I3_14]["dir"]
        sub_directory = available_datasets[I3_14]["subdir"]

        try:
            np.loadtxt(
                os.path.join(
                    data_directory,
                    directory,
                    sub_directory,
                    "uptime",
                    "IC40_exp.csv",
                )
            )
        except FileNotFoundError:
            data_interface = IceCubeData()
            data_interface.fetch(I3_14)

        energy = []
        ra = []
        dec = []
        mjd = []
        ang_err = []
        type = []

        def _append_data(s):
            data = np.loadtxt(
                join(
                    data_directory,
                    f"{str(Path(directory) / Path(sub_directory))}/events/{s}_exp.csv",
                )
            )

            energy.append(data[:, cls.energy_])
            mjd.append(data[:, cls.mjd_])
            ra.append(data[:, cls.ra_])
            dec.append(data[:, cls.dec_])
            ang_err.append(data[:, cls.ang_err_])
            type.append(len(data[:, cls.energy_]) * [s])

        for s in seasons:
            if s == IC86:
                for suffering in suffixes:
                    _append_data(str(s) + suffering)
            else:
                _append_data(s)

        energy = np.power(10, np.concatenate(energy)) << u.GeV
        mjd = Time(np.concatenate(mjd), format="mjd")
        ra = np.concatenate(ra) << u.deg
        dec = np.concatenate(dec) << u.deg
        ang_err = np.concatenate(ang_err) << u.deg
        type = np.concatenate(type)
        coord = SkyCoord(ra=ra, dec=dec, frame="icrs")

        events = cls(energy, coord, type, ang_err, mjd)
        events._ra = ra
        events._dec = dec

        return events
