"""
Classes to organise the effective area of the IceCube detector for track events
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
from astropy import units as u
from pathlib import Path
from typing import Self


from icecube_data_reader.event_types import Refrigerator, EventType
from icecube_data_reader.downloader import available_datasets, data_directory, I3_14

from astropy import units as u
from astropy.units.typing import QuantityLike


class EffectiveArea(ABC):
    
    
    @classmethod
    @abstractmethod
    def load(cls):
        """Load effective area object"""

        pass

    @property
    def eff_area(self):
        """
        2D histogram of effective area values,
        Etrue on axis 0 and cosz on axis 1
        """

        return self._eff_area

    @property
    def cosz_bin_edges(self):
        """
        cos(zenith) bin edges corresponding to the
        histogram in eff_area.
        """

        return self._cosz_bin_edges
    
    #@property
    #def sin_dec_bin_edges(self):
    #    """
    #    sin(dec) bin edges corresponding to the
    #    histogram in eff_area
    #    """
    #
    #    return self._sin_dec_bin_edges    
    
    @property
    def tE_bin_edges(self):
        """
        True energy bin edges corresponding the the
        histogram in eff_area.
        """

        return self._tE_bin_edges


class IceTrackDR2EffectiveArea(EffectiveArea):

    @u.quantity_input
    def __init__(
            self,
            eff_area: u.m**2,
            tE_bin_edges: u.GeV,
            dec_bin_edges: u.rad,
            season: EventType,
            ):
        
        self._season = season
        self._tE_bin_edges = tE_bin_edges
        # this inverts the order of bins
        self._cosz_bin_edges = np.sort(- np.sin(dec_bin_edges.to_value(u.rad)))
        # hence we invert the order of elements along the cosz axis of eff_area
        self._eff_area = np.flip(eff_area, axis=1)
        #self._sin_dec_bin_edges = np.sin(dec_bin_edges.to_value(u.rad))

    @classmethod
    def load(cls, season: EventType, data_path: Path = data_directory) -> Self:
        """Create effective area object

        :param season: Detector season of
        :type season: EventType
        :param data_path: Path to lookf or data files, defaults to data_directory
        :type data_path: Path, optional
        :return: Effective area instance
        :rtype: :py:class:`icecube_data_reader.irf.effective_area.IceTrackDR2EffectiveArea`
        """

        season_string = str(season)
        file = data_path / Path(
            available_datasets[I3_14]["dir"]
        ) / Path(
        available_datasets[I3_14]["subdir"]
        ) / Path(
            f"irfs/{season_string}_effectiveArea.csv"
        )

        filelayout = ["Emin", "Emax", "DECmin", "DECmax", "Aeff"]

        output = pd.read_csv(file, comment="#", sep="\s+", names=filelayout).to_dict()

        true_energy_lower = np.array(list(set(output["Emin"].values())))
        true_energy_upper = np.array(list(set(output["Emax"].values())))


        dec_lower = np.array(list(set(output["DECmin"].values())))
        dec_upper = np.array(list(set(output["DECmax"].values())))

        tE_bin_edges = np.sort(np.power(10, 
            np.unique(np.vstack((true_energy_lower, true_energy_upper)))
        )) << u.GeV

        dec_bin_edges = np.sort(np.radians(np.unique(np.vstack((dec_lower, dec_upper))))) << u.rad

        eff_area = np.reshape(
            np.array(list(output["Aeff"].values())) * 1e-4,
            (len(dec_lower), len(true_energy_lower)),
        ).T << u.m**2

        aeff = cls(eff_area, tE_bin_edges, dec_bin_edges, season)

        return aeff
