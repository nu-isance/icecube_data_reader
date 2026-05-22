"""
Organise lifetime intervals during which IceCube is operational and taking data
"""

import numpy as np
from scipy import stats
import os
from abc import ABC
from astropy import units as u
from icecube_data_reader.downloader import data_directory, I3_14, available_datasets, IceCubeData
from icecube_data_reader.event_types import IC40, IC59, IC79, IC86, suffixes, EventType

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LifeTime(ABC):
    @property
    def data(self):
        return self._data

    def lifetime_from_mjd(
        self, mjd_min: float, mjd_max: float, squeeze: bool = True
    ) -> dict[EventType | u.quantity.Quantity[u.yr]]:
        """Compute lifetime from provided min and max mjd.

        :param mjd_min: Minimum mjd
        :type mjd_min: float
        :param mjd_max: Maximum mjd
        :type mjd_max: float
        :param squeeze: If true, remove keys with 0 duration
        :type squeeze: bool
        :return: Dictionary with detector season: duration in astropy.units.yr
        :rtype: dict[EventType|u.quantity.Quantity[u.yr]]
        """

        if mjd_min < self._times[0, 0]:
            logger.warning(f"{mjd_min} is outside of experiment.")

        if mjd_max > self._times[-1, -1]:
            logger.warning(f"{mjd_max} is outside of experiment.")

        output = {}
        for s in self._data.keys():
            # Query histograms for fraction of total lifetime in a season
            # multiply by total lifetime in a season to get appropriate value
            time = (self._dists[s].cdf(mjd_max) - self._dists[s].cdf(mjd_min)) * self._lifetimes[s]
            # set atol to 1e-9 days, so we are below the time resolution of event mjd (1e-8 days)
            if squeeze and np.isclose(time.to_value(u.d), 0.0, atol=1e-9):
                time = 0 * u.yr

            output[s] = time

        return output

    def lifetime_from_season(self, *seasons) -> dict[EventType | u.quantity.Quantity[u.yr]]:
        """Compute lifetime of given seasons

        :param seasons: Seasons to calculate lifetimes of
        :return: Dictionary with detector season: duration in astropy.units.yr
        :rtype: dict[EventType|u.quantity.Quantity[u.yr]]
        """

        output = {}
        for s in seasons:
            output[s] = self._lifetimes[s]

        return output


class DR2LifeTime(LifeTime):
    def __init__(self):

        directory = available_datasets[I3_14]["dir"]
        sub_directory = available_datasets[I3_14]["subdir"]
        # Check if data exists, otherwise download it
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

        # Array of start and end times
        self._times = np.zeros((4, 2))
        self._data = {}
        self._dists = {}
        self._lifetimes = {}
        # Dict keeps this order of keys
        for c, s in enumerate([IC40, IC59, IC79, IC86]):
            if s != IC86:
                self._data[s] = np.loadtxt(
                    os.path.join(
                        data_directory,
                        directory,
                        sub_directory,
                        "uptime",
                        f"{s}_exp.csv",
                    )
                )
            else:
                data = []
                for suffering in suffixes:
                    data.append(
                        np.loadtxt(
                            os.path.join(
                                data_directory,
                                directory,
                                sub_directory,
                                "uptime",
                                f"{str(s) + suffering}_exp.csv",
                            )
                        )
                    )
                self._data[s] = np.concatenate(data)
            # Store start and end times of each season
            self._times[c, 0] = self._data[s][0, 0]
            self._times[c, 1] = self._data[s][-1, -1]
            self._lifetimes[s] = (np.sum(np.diff(self._data[s])) << u.d).to(u.yr)
            # Create histogram of on/off data
            # bins are just increasing values of the flattened times
            bins = np.sort(self._data[s].flatten())
            on_off = np.zeros(bins.size - 1)
            # Every other entry is a one, indicating running detector
            on_off[::2] = 1.0
            # density=True for on_off to be treated as density
            self._dists[s] = stats.rv_histogram((on_off, bins), density=True)
