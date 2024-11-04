# -*- coding: utf-8 -*-
""" Class handling the information describing the estimations done on a single location.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 11 sep 2021
"""
import warnings
from enum import IntEnum
from typing import List, Optional, Union

import numpy as np
from shapely.geometry import Point

from ..waves_exceptions import WavesEstimationAttributeError
from .bathymetry_sample_estimation import BathymetrySampleEstimation


class SampleStatus(IntEnum):
    """ Enum specifying the synthetic status which can be given to a point in the product."""
    SUCCESS = 0
    FAIL = 1
    ON_GROUND = 2
    NO_DATA = 3
    NO_DELTA_TIME = 4
    OUTSIDE_ROI = 5
    BEYOND_OFFSHORE_LIMIT = 6


class BathymetrySampleEstimations(list):
    """ This class gathers information relevant to some location, whatever the bathymetry
    estimators, as well as a list of bathymetry estimations made at this location.
    """
# TODO: add a keep_only() method to reduce the list to a maximum number of estimations.

    def __init__(self, location: Point, gravity: float, delta_time: float,
                 distance_to_shore: float, inside_roi: bool, inside_offshore_limit: bool) -> None:
        super().__init__()

        self._location = location
        self._gravity = gravity
        self._distance_to_shore = distance_to_shore
        self._inside_roi = inside_roi
        self._inside_offshore_limit = inside_offshore_limit
        self._delta_time = delta_time

        self._data_available = True
        self._delta_time_available = True
        self._physical_wave_fields_available = True

    def append(self, estimation: BathymetrySampleEstimation) -> None:
        """ Store a single estimation into the estimations list, ensuring that there are no
        duplicate estimations for the same (direction, wavelength) pair.

        :param estimation: a new estimation to store inside this localized list of estimations
        """
        stored_estimations_hashes = [hash(estim) for estim in self]
        # Do not store duplicate estimations for the same direction/wavelength
        if hash(estimation) in stored_estimations_hashes:
            warnings.warn(f'\nTrying to store a duplicate estimation:\n{str(estimation)} ')
        else:
            super().append(estimation)

    def sort_on_attribute(self, attribute_name: Optional[str] = None, reverse: bool = True) -> None:
        """ Sort in place the wave fields estimations based on one of their attributes.

        :param attribute_name: name of an attribute present in all estimations to use for sorting
        :param reverse: When True sorting is in descending order, when False in ascending order
        """
        if attribute_name is not None:
            name = attribute_name
            self.sort(key=lambda x: getattr(x, name), reverse=reverse)

    def argsort_on_attribute(self, attribute_name: Optional[str] = None,
                             reverse: bool = True) -> List[int]:
        """ Return the indices of the wave fields estimations which would sort them based
        on one of their attributes.

        :param attribute_name: name of an attribute present in all estimations to use for sorting
        :param reverse: When True sorting is in descending order, when False in ascending order
        :returns: either en empty list if attribute_name is None or the list of indices which would
                  sort this BathymetrySampleEstimations according to one of the attributes.
        """
        if attribute_name is not None:
            attr_list = [getattr(estimation, attribute_name) for estimation in self]
            arg_sorted = np.argsort(attr_list).tolist()
            if reverse:
                arg_sorted.reverse()
            return arg_sorted
        return []

    def get_attribute(self, attribute_name: str) -> Union[float, List[float]]:
        """ Retrieve the values of an attribute either at the level of BathymetrySampleEstimations
        or in the list of BathymetrySampleEstimation instances

        :param attribute_name: name of the estimation attribute to retrieve
        :returns: the values of the attribute either as a scalar or a list of values
        :raises WavesEstimationAttributeError: when the attribute does not exist
        """
        # Firstly try to find the attribute from the estimations common attributes
        if hasattr(self, attribute_name):
            # retrieve attribute from the estimations header
            bathymetry_estimation_attribute = getattr(self, attribute_name)
        else:
            if not self:
                err_msg = f'Attribute {attribute_name} undefined (no estimations)'
                raise WavesEstimationAttributeError(err_msg)
            bathymetry_estimation_attribute = self.get_estimations_attribute(attribute_name)
        return bathymetry_estimation_attribute

    def get_estimations_attribute(self, attribute_name: str) -> List[float]:
        """ Retrieve the values of some attribute in the list of stored wave field estimations.

        :param attribute_name: name of the attribute to retrieve
        :returns: the values of the attribute in the order where the estimations are stored
        :raises WavesEstimationAttributeError: when the attribute does not exist in at least
                                               one estimation
        """
        try:
            return [getattr(estimation, attribute_name) for estimation in self]
        except AttributeError as excp:
            err_msg = f'Attribute {attribute_name} undefined for some wave field estimation'
            raise WavesEstimationAttributeError(err_msg) from excp

    def remove_unphysical_wave_fields(self) -> None:
        """  Remove unphysical wave fields
        """
        # If a physical estimation is in the list, the flag is positionned
        physical_mask = [e.is_physical() for e in self]
        self._physical_wave_fields_available = True in physical_mask

    @property
    def location(self) -> Point:
        """ :returns: The (X, Y) coordinates of this estimation location"""
        return self._location

    @property
    def distance_to_shore(self) -> float:
        """ :returns: The distance from this estimation location to the nearest shore (km)"""
        return self._distance_to_shore

    @property
    def inside_roi(self) -> bool:
        """ :returns: True if the point is inside the defined ROI, False otherwise"""
        return self._inside_roi

    @property
    def inside_offshore_limit(self) -> bool:
        """ :returns: True if the distance to shore is inferior or equal to the offshore limit,
                      False otherwise"""
        return self._inside_offshore_limit

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at this estimation location (m/s2)
        """
        return self._gravity

    @property
    def delta_time(self) -> float:
        """ :returns: the time difference between the images used for this estimation """
        return self._delta_time

    @delta_time.setter
    def delta_time(self, value: bool) -> None:
        self._delta_time = value

    @property
    def delta_time_available(self) -> bool:
        """ :returns: True if delta time was available for doing estimations, False otherwise """
        return not np.isnan(self.delta_time)

    @property
    def data_available(self) -> bool:
        """ :returns: True if data was available for doing the estimations, False otherwise """
        return self._data_available

    @data_available.setter
    def data_available(self, value: bool) -> None:
        self._data_available = value

    @property
    def status(self) -> int:
        """ :returns: a synthetic value giving the final estimation status
        """
        status = SampleStatus.SUCCESS
        if self.distance_to_shore <= 0.:
            status = SampleStatus.ON_GROUND
        elif not self.inside_offshore_limit:
            status = SampleStatus.BEYOND_OFFSHORE_LIMIT
        elif not self.inside_roi:
            status = SampleStatus.OUTSIDE_ROI
        elif not self.data_available:
            status = SampleStatus.NO_DATA
        elif not self.delta_time_available:
            status = SampleStatus.NO_DELTA_TIME
        elif not self._physical_wave_fields_available:
            status = SampleStatus.FAIL
        return status.value

    def __str__(self) -> str:
        result = f'+++++++++ Set of estimations made at: {self.location} \n'
        result += f'  distance to shore: {self.distance_to_shore}   gravity: {self.gravity}\n'
        result += '  availability: '
        result += f' (data: {self.data_available}, delta time: {self.delta_time_available})\n'
        result += f'  STATUS: {self.status}'
        result += ' (0: SUCCESS, 1: FAIL, 2: ON_GROUND, 3: NO_DATA, 4: NO_DELTA_TIME,'
        result += ' 5: OUTSIDE_ROI, 6: BEYOND_OFFSHORE_LIMIT)\n'
        result += f'{len(self)} estimations available:\n'
        for index, estimation in enumerate(self):
            result += f'---- estimation {index} ---- type: {type(estimation).__name__}\n'
            result += str(estimation) + '\n'
        return result
