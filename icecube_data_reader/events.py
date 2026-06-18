"""
Class to organise IceCube event data
"""

from abc import ABC, abstractmethod
import os
from os.path import join
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Iterable
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time
import h5py
from time import time as thyme

from icecube_data_reader.downloader import (
    data_directory,
    I3_14,
    available_datasets,
    IceCubeData,
)
from icecube_data_reader.event_types import (
    IC40,
    IC59,
    IC79,
    IC86,
    suffixes,
    EventType,
    Refrigerator,
)
from icecube_data_reader.lifetime import LifeTime

from typing import Self
import logging

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
    def energies(self):
        return self._energies

    @property
    def ang_errs(self):
        return self._ang_errs

    @property
    def coords(self):
        return self._coords

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def unit_vectors(self):
        return self._unit_vectors

    @property
    def types(self):
        return self._types

    @property
    def int_types(self):
        return self._int_types

    @property
    def N(self):
        return self.types.size

    @u.quantity_input
    def apply_energy_cut(self, Emin: u.GeV, Emax: u.GeV = np.inf * u.GeV):
        """Select events based on (reconstructed) energy

        :param Emin: Minimum energy
        :type Emin: u.GeV
        :param Emax: Maximum energy, defaults to np.inf*u.GeV
        :type Emax: u.GeV, optional
        """
        pass

    @classmethod
    @abstractmethod
    def from_file():
        pass

    @abstractmethod
    def to_file():
        pass

    @abstractmethod
    def remove(self, i) -> None:
        pass


class IceTrackDR2Events(Events):
    """
    Organise event data of the IceTracks-DR2
    """

    mjd_ = 3
    energies_ = 4
    ang_errs_ = 5
    ras_ = 6
    decs_ = 7

    @u.quantity_input
    def __init__(
        self,
        energies: u.GeV,
        coords: SkyCoord,
        types: Iterable[EventType],
        ang_errs: u.deg,
        mjd: Time,
    ):
        self._energies = energies
        self._coords = coords
        self._types = types
        self._int_types = np.array([_.S for _ in types])
        self._ang_errs = ang_errs
        self._mjd = mjd
        self._coords.representation_type = "cartesian"
        self._unit_vectors = np.array(
            [coords.x.value, coords.y.value, coords.z.value]
        ).T
        self._coords.representation_type = "spherical"

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

    def select(self, mask: npt.NDArray[np.bool_]) -> None:
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

        self._energies = self._energies[mask]
        self._coords = self._coords[mask]
        self._unit_vectors = self._unit_vectors[mask]
        self._types = self._types[mask]
        self._int_types = self._int_types[mask]
        self._ang_errs = self._ang_errs[mask]
        self._mjd = self._mjd[mask]

    def remove(self, i: int) -> None:
        """
        Remove the event at index i
        :param i: Event index
        :type i: int
        """
        self._energies = np.delete(self._energies, i)
        self._coords = np.delete(self._coords, i)
        self._unit_vectors = np.delete(self._unit_vectors, i, axis=0)
        self._types = np.delete(self._types, i)
        self._ang_errs = np.delete(self._ang_errs, i)
        self._mjd = np.delete(self._mjd, i)
        self._int_types = np.delete(self._int_types, i)

    def to_file(
        self,
        path: Path,
        append: bool = False,
        group_name: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write events to file.
        Keyworded arguments control behaviour with existing files.
        If not overwrite, but `path` exists, append a timestamp to the file name.

        :param path: File path
        :type path: Path
        :param append: If true, append to existing file, defaults to False
        :type append: bool, optional
        :param group_name: If provided, create new group in path and write events there, defaults to None
        :type group_name: str | None, optional
        :param overwrite: If true, overwrite existing path, defaults to False
        :type overwrite: bool, optional

        :return: path Object
        :rtype: Path
        """

        self._file_keys = ["energies", "unit_vectors", "types", "ang_errs", "mjd"]
        self._file_values = [
            self.energies.to(u.GeV).value,
            self.unit_vectors,
            self.int_types,
            self.ang_errs.to(u.deg).value,
            self.mjd.mjd,
        ]

        if append:
            with h5py.File(path, "r+") as f:
                if group_name is None:
                    event_folder = f.create_group("events")
                else:
                    event_folder = f.create_group(group_name)

                for key, value in zip(self._file_keys, self._file_values):
                    event_folder.create_dataset(key, data=value)

        else:
            dirname = os.path.dirname(path)
            filename = os.path.basename(path)
            if dirname:
                if not os.path.exists(dirname):
                    logger.warning(
                        f"{dirname} does not exist, saving instead to {os.getcwd()}"
                    )
                    dirname = os.getcwd()
            else:
                dirname = os.getcwd()
            path = Path(dirname) / Path(filename)
            if os.path.exists(filename) and not overwrite:
                logger.warning(f"File {filename} already exists.")
                file = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                file += f"_{int(thyme())}"
                filename = file + ext

            path = Path(dirname) / Path(filename)

            with h5py.File(path, "w") as f:
                if group_name is None:
                    event_folder = f.create_group("events")
                else:
                    event_folder = f.create_group(group_name)

                for key, value in zip(self._file_keys, self._file_values):
                    event_folder.create_dataset(key, data=value)
        return path

    @classmethod
    def from_file(
        cls,
        filename: Path,
        group_name: str = None,
    ) -> Self:
        """Load events from .h5 file

        :param filename: File to load events from
        :type filename: Path
        :param group_name: Name of events group, if provided when writing to file, defaults to None
        :type group_name: str, optional
        """
        with h5py.File(filename, "r") as f:
            if group_name is None:
                events_folder = f["events"]
            else:
                events_folder = f[group_name]

            energies = events_folder["energies"][()] * u.GeV
            uv = events_folder["unit_vectors"][()]
            int_types = events_folder["types"][()]
            types = np.array([Refrigerator.int2dm(_) for _ in int_types])
            ang_errs = events_folder["ang_errs"][()] * u.deg

            # For backwards compatibility
            try:
                mjd = events_folder["mjd"][()]
            except KeyError:
                mjd = [99.0] * len(energies)

        coords = SkyCoord(
            uv.T[0], uv.T[1], uv.T[2], representation_type="cartesian", frame="icrs"
        )
        mjd = Time(mjd, format="mjd")
        coords.representation_type = "spherical"
        events = cls(energies, coords, types, ang_errs, mjd)

        return events

    @u.quantity_input
    def apply_energy_cut(self, Emin: u.GeV, Emax=np.inf * u.GeV) -> None:
        """Apply energy cuts to events

        :param Emin: Minimum allowed energy
        :type Emin: u.GeV
        :param Emax: Maximum allowed energy, defaults to np.inf*u.GeV
        :type Emax: u.GeV, optional
        """
        mask = (self.energies >= Emin) & (self.energies <= Emax)
        self.select(mask)

    @classmethod
    def from_event_files(cls, *seasons: EventType | str) -> Self:
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

        energies = []
        ras = []
        decs = []
        mjd = []
        ang_errs = []
        types = []

        def _append_data(s, suffering: str = ""):
            data = np.loadtxt(
                join(
                    data_directory,
                    f"{str(Path(directory) / Path(sub_directory))}/events/{str(s)+suffering}_exp.csv",
                )
            )

            energies.append(data[:, cls.energies_])
            mjd.append(data[:, cls.mjd_])
            ras.append(data[:, cls.ras_])
            decs.append(data[:, cls.decs_])
            ang_errs.append(data[:, cls.ang_errs_])
            types.append(len(data[:, cls.energies_]) * [s])

        for s in seasons:
            s = Refrigerator.str2dm(s) if isinstance(s, str) else s
            if s == IC86:
                for suffering in suffixes:
                    _append_data(s, suffering)
            else:
                _append_data(s)

        energies = np.power(10, np.concatenate(energies)) << u.GeV
        mjd = Time(np.concatenate(mjd), format="mjd")
        ras = np.concatenate(ras) << u.deg
        decs = np.concatenate(decs) << u.deg
        ang_errs = np.concatenate(ang_errs) << u.deg
        types = np.concatenate(types)
        coords = SkyCoord(ra=ras, dec=decs, frame="icrs")

        events = cls(energies, coords, types, ang_errs, mjd)
        events._ras = ras
        events._decs = decs

        return events
