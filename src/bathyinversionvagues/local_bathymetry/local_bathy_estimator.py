# -*- coding: utf-8 -*-
""" Base class for the estimators of waves fields from several images taken at a small
time intervals.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from abc import abstractmethod, ABC
from copy import deepcopy

from typing import Dict, Any, List, Optional, TYPE_CHECKING  # @NoMove

import numpy as np

from ..image.image_geometry_types import PointType
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from .waves_field_estimation import WavesFieldEstimation
# from .waves_fields_estimations import WavesFieldsEstimations


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport

# TODO: create a true class encapsulating the estimations and providing room for scalar infos
# (distoshore, gravity, delta_time.) as well as logics for handling dimensions.
WavesFieldsEstimationsList = List[WavesFieldEstimation]


class LocalBathyEstimator(ABC):
    """ Abstract base class of all local bathymetry estimators.
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param images_sequence: a list of superimposed local images centered around the position
                                where the estimator is working.
        :param global_estimator: a global bathymetry estimator able to provide the services needed
                                 by this local bathymetry estimator (access to parameters,
                                 data providers, debugging, ...)
        :param selected_directions: the set of directions onto which the sinogram must be computed
        """
        # TODO: Check that the images have the same resolution, satellite (and same size ?)
        self.global_estimator = global_estimator
        self.debug_sample = self.global_estimator.debug_sample
        self.local_estimator_params = self.global_estimator.waveparams

        self.images_sequence = images_sequence
        self.selected_directions = selected_directions

        self._position = (0., 0.)
        self._gravity = 0.
        self._waves_fields_estimations: WavesFieldsEstimationsList = []

        self._delta_time = 0.

        self._metrics: Dict[str, Any] = {}

    def set_position(self, point: PointType) -> None:
        """ Specify the cartographic position where this local bathy estimator is working and
        updates the localized data (gravity, delta time) accordingly

        :param point: a tuple (X, Y) of cartographic coordinates
        """
        self._position = point
        self._gravity = self.global_estimator.get_gravity(self._position, 0.)
        self._delta_time = self.global_estimator.get_delta_time(self._position)

    @property
    @abstractmethod
    def preprocessing_filters(self) -> ImageProcessingFilters:
        """ :returns: A list of functions together with their parameters to be applied
        sequentially to all the images of the sequence before subsequent bathymetry estimation.
        """

    def preprocess_images(self) -> None:
        """ Process the images before doing the bathymetry estimation with a sequence of
        image processing filters.
        """
        for image in self.images_sequence:
            image.apply_filters(self.preprocessing_filters)

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at the working position of the estimator
        """
        return self._gravity

    # FIXME: At the moment only a pair of images is handled (list is limited to a singleton)
    @property
    def delta_time(self) -> float:
        """ :returns: the time differences between 2 consecutive frames in the image sequence
        """
        return self._delta_time

    @abstractmethod
    def run(self) -> None:
        """  Run the local bathymetry estimation, using some method specific to the inheriting
        class.

        This method stores its results using the store_estimation() method and
        its metrics in _metrics attribute.
        """

    @abstractmethod
    def sort_waves_fields(self) -> None:
        """  Sorts the waves fields on whatever criteria.
        """

    def validate_waves_fields(self) -> None:
        """  Remove non physical waves fields
        """
        # Filter non physical waves fields and bathy estimations
        filtered_out_waves_fields = [
            field for field in self.waves_fields_estimations if
            field.period >= self.local_estimator_params.MIN_T and
            field.period <= self.local_estimator_params.MAX_T]
        filtered_out_waves_fields = [
            field for field in filtered_out_waves_fields if
            field.linearity >= self.local_estimator_params.MIN_WAVES_LINEARITY and
            field.linearity <= self.local_estimator_params.MAX_WAVES_LINEARITY]
        self.waves_fields_estimations = filtered_out_waves_fields

    @abstractmethod
    def create_waves_field_estimation(self, direction: float, wavelength: float
                                      ) -> WavesFieldEstimation:
        """ Creates the WavesFieldEstimation instance where the local estimator will store its
        estimations.

        :param direction: the propagation direction of the waves field (degrees measured clockwise
                          from the North).
        :param wavelength: the wavelength of the waves field
        :returns: an initialized instance of WavesFilesEstimation to be filled in further on.
        """

    def store_estimation(self, waves_field_estimation: WavesFieldEstimation) -> None:
        """ Store a single estimation into the estimations list

        :param waves_field_estimation: a new estimation to store for this local bathy estimator
        """
        if self._waves_fields_estimations is None:
            raise ValueError('No waves_fields_estimations defined attribute yet')
        self._waves_fields_estimations.append(waves_field_estimation)

    @property
    def waves_fields_estimations(self) -> WavesFieldsEstimationsList:
        """ :returns: a copy of the estimations recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        if self._waves_fields_estimations is None:
            raise ValueError('No waves_fields_estimations defined attribute yet')
        return deepcopy(self._waves_fields_estimations)

    @waves_fields_estimations.setter
    def waves_fields_estimations(self, estimations: WavesFieldsEstimationsList) -> None:
        self._waves_fields_estimations = estimations

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a copy of the dictionary of metrics recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return deepcopy(self._metrics)

    def print_estimations_debug(self, step: str) -> None:
        """ Print debugging info on the estimations if the point has been tagged for debugging

        :param step: A string to be printed as header of the debugging info.
        """
        self.global_estimator.print_estimations_debug(self.waves_fields_estimations, step)


class LocalBathyEstimatorDebug(LocalBathyEstimator):

    def run(self) -> None:
        super().run()
        if self.debug_sample:
            self.draw_results()

    @abstractmethod
    def draw_results(self) -> None:
        """ Save a diagram to help comprehension about result on current point
        """
