# -*- coding: utf-8 -*-
""" Base class for the estimators of wave fields from several images taken at a small
time intervals.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from abc import abstractmethod, ABC
from copy import deepcopy

from typing import Dict, Any, Optional, Type, TYPE_CHECKING  # @NoMove

import numpy as np

from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation
from ..data_model.bathymetry_sample_estimations import BathymetrySampleEstimations
from ..image.image_geometry_types import PointType
from ..image_processing.images_sequence import ImagesSequence
from ..image_processing.waves_image import ImageProcessingFilters
from ..waves_exceptions import SequenceImagesError, WavesEstimationError


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class LocalBathyEstimator(ABC):
    """ Abstract base class of all local bathymetry estimators.
    """

    final_estimations_sorting: Optional[str] = None

    @property
    @classmethod
    @abstractmethod
    def wave_field_estimation_cls(cls) -> Type[BathymetrySampleEstimation]:
        """ :returns: a class inheriting from BathymetrySampleEstimation to use for storing an
                      estimation.
        """

    def __init__(self, location: PointType, images_sequence: ImagesSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ Constructor

        :param location: The (X, Y) coordinates of the location where this estimator is acting
        :param images_sequence: a list of superimposed local images centered around the position
                                where the estimator is working.
        :param global_estimator: a global bathymetry estimator able to provide the services needed
                                 by this local bathymetry estimator (access to parameters,
                                 data providers, debugging, ...)
        :param selected_directions: the set of directions onto which the sinogram must be computed
        :raises SequenceImagesError: when the sequence is empty
        """
        if not images_sequence:
            raise SequenceImagesError('Sequence images is empty')
        self.images_sequence = images_sequence
        self.spatial_resolution = images_sequence.resolution

        self.global_estimator = global_estimator
        self.debug_sample = self.global_estimator.debug_sample
        self.local_estimator_params = self.global_estimator.local_estimator_params

        self.selected_directions = selected_directions

        # FIXME: distance to shore test should take into account windows sizes
        distance = self.global_estimator.get_distoshore(location)
        gravity = self.global_estimator.get_gravity(location, 0.)
        inside_roi = self.global_estimator.is_inside_roi(location)

        self._set_sequential_delta_times(location)

        self._bathymetry_estimations = BathymetrySampleEstimations(location, gravity,
                                                                   self.propagation_duration,
                                                                   distance, inside_roi)

        self._metrics: Dict[str, Any] = {}

    def _set_sequential_delta_times(self, location: PointType) -> None:
        """ Computes the list of time differences between 2 consecutive frames in the image sequence
        """
        sequential_delta_times = []
        for frame_index in range(len(self.images_sequence) - 1):
            delta_time = self.global_estimator.get_delta_time(
                self.images_sequence._images_id[frame_index],
                self.images_sequence._images_id[frame_index + 1],
                location)
            sequential_delta_times.append(delta_time)
        self._sequential_delta_times = np.array(sequential_delta_times)

    def can_estimate_bathy(self) -> bool:
        return (self.bathymetry_estimations.distance_to_shore > 0 and
                self.bathymetry_estimations.inside_roi)

    @property
    @abstractmethod
    def nb_used_frames(self) -> int:
        """ :returns: The number of frames used by an estimator.
        """

    @property
    def propagation_duration(self) -> float:
        """ :returns: The time length of the sequence of images used for the estimation. May be
                      positive or negative to account for chronology of start and stop images.
        """
        if self.nb_used_frames > len(self.images_sequence):
            raise WavesEstimationError(
                'The chosen number of lag frames is greater than the number of available frames')
        # FIXME: this slicing is wrong when frames are not the first ones in the sequence
        return np.sum(self._sequential_delta_times[:self.nb_used_frames - 1])

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
            filtered_image = image.apply_filters(self.preprocessing_filters)
            image.pixels = filtered_image.pixels

    @property
    def gravity(self) -> float:
        """ :returns: the acceleration of the gravity at the working position of the estimator
        """
        return self.bathymetry_estimations.gravity

    @property
    def location(self) -> PointType:
        """ :returns: The (X, Y) coordinates of the location where this estimator is acting"""
        return self.bathymetry_estimations.location

    @abstractmethod
    def run(self) -> None:
        """  Run the local bathymetry estimation, using some method specific to the inheriting
        class.

        This method stores its results in the bathymetry_estimations list and
        its metrics in _metrics attribute.
        """

    def create_bathymetry_estimation(self, direction: float, wavelength: float
                                     ) -> BathymetrySampleEstimation:
        """ Creates the BathymetrySampleEstimation instance where the local estimator will store its
        estimation.

        :param direction: the propagation direction of the wave field (degrees measured
                          counterclockwise from the East).
        :param wavelength: the wavelength of the wave field
        :returns: an initialized instance of BathymetrySampleEstimation to be filled in further on.
        """
        bathy_estimation = self.wave_field_estimation_cls(
            self.gravity,
            self.global_estimator.depth_estimation_method,
            self.global_estimator.waves_period_range,
            self.global_estimator.waves_linearity_range,
            self.global_estimator.depth_min)
        bathy_estimation.delta_time = self.propagation_duration
        bathy_estimation.direction = direction
        bathy_estimation.wavelength = wavelength

        return bathy_estimation

    @property
    def bathymetry_estimations(self) -> BathymetrySampleEstimations:
        """ :returns: the wave fields estimations recorded by this estimator.
        """
        return self._bathymetry_estimations

    @property
    def metrics(self) -> Dict[str, Any]:
        """ :returns: a copy of the dictionary of metrics recorded by this estimator.
                      Used for freeing references to memory expensive data (images, transform, ...)
        """
        return self._metrics

    @metrics.setter
    def metrics(self, values: Dict[str, Any]) -> None:
        self._metrics = deepcopy(values)
