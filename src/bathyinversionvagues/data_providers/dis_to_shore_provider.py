# -*- coding: utf-8 -*-
""" Definition of the DistToShoreProvider abstract class

:author: GIROS Alain
:created: 23/06/2021
"""
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, Any  # @NoMove

from shapely.geometry import Point

import xarray as xr  # @NoMove
import numpy as np

from .localized_data_provider import LocalizedDataProvider


class DisToShoreProvider(ABC, LocalizedDataProvider):
    """ A distoshore provider is a service which is able to provide the distance to shore of a
    point specified by its coordinates in some SRS.
    """

    @abstractmethod
    def get_distoshore(self, point: Point) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a point expressed in the SRS coordinates set for this provider
        :returns: the distance to shore in kilometers (positive over water, zero on ground,
                  positive infinity if unknown).
        """


class InfinityDisToShoreProvider(DisToShoreProvider):
    """ A DistToShoreProvider which provides infinity distance to any request, ensuring that any
    point is always considered on water.
    """

    def get_distoshore(self, point: Point) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a point expressed in the SRS coordinates set for this provider
        :returns: positive infinity for any position
        """
        _ = point
        return np.Infinity


class NetCDFDisToShoreProvider(DisToShoreProvider):
    """ A DistToShoreProvider which provides the distance to shore when it is stored in a
    'disToShore' layer of a netCDF file.
    """

    # FIXME: EPSG code needed because no SRS retrieved from the NetCDF file at this time.
    def __init__(self, distoshore_file_path: Path, distoshore_epsg_code: int,
                 x_axis_label: str, y_axis_label: str) -> None:
        """ Create a NetCDFDisToShoreProvider object and set necessary informations

        :param distoshore_file_path: full path of a netCDF file containing the distance to shore
                                     to be used by this provider.
        :param distoshore_epsg_code: the EPSG code of the SRS used in the NetCDF file.
        :param x_axis_label: Label of the x axis of the dataset ('x' or 'lon' for instance)
        :param y_axis_label: Label of the y axis of the dataset ('y' or 'lat' for instance)
        """
        super().__init__()
        self.provider_epsg_code = distoshore_epsg_code
        self._x_axis_label = x_axis_label
        self._y_axis_label = y_axis_label

        self._distoshore_file_path = distoshore_file_path
        self._distoshore_xarray: Optional[Any] = None

    def get_distoshore(self, point: Point) -> float:
        """ Provides the distance to shore of a point in kilometers.

        :param point: a point expressed in the SRS coordinates set for this provider
        :returns: the distance to the nearest shore (km)
        """
        if self._distoshore_xarray is None:
            self._distoshore_xarray = xr.open_dataset(self._distoshore_file_path)
        provider_point = self.transform_point((point.x, point.y), 0.)
        kw_sel = {self._x_axis_label: provider_point[0],
                  self._y_axis_label: provider_point[1],
                  'method': 'nearest'}
        distance_xr_dataset = self._distoshore_xarray.sel(**kw_sel)
        return float(distance_xr_dataset['disToShore'].values)
