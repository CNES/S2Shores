# -*- coding: utf-8 -*-
""" Abstract Class offering a common template for temporal correlation method and spatial
correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from abc import abstractmethod
from typing import Optional, List, TYPE_CHECKING  # @NoMove

from munch import Munch
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt

import numpy as np

from ..generic_utils.image_filters import detrend, clipping
from ..generic_utils.signal_filters import filter_mean, remove_median
from ..generic_utils.signal_utils import find_period, find_dephasing
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, SignalProcessingFilters
from ..image_processing.waves_sinogram import WavesSinogram

from .correlation_waves_field_estimation import CorrelationWavesFieldEstimation
from .local_bathy_estimator import LocalBathyEstimator


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class CorrelationBathyEstimator(LocalBathyEstimator):
    """ Class offering a framework for bathymetry computation based on correlation
    """

    def __init__(self, images_sequence: List[WavesImage], global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        """ constructor
        :param images_sequence: sequence of image used to compute bathymetry
        :param global_estimator: global estimator
        :param selected_directions: selected_directions: the set of directions onto which the
        sinogram must be computed
        """
        super().__init__(images_sequence, global_estimator, selected_directions)
        # Physical attributes
        self._direction_propagation: Optional[float] = None
        self._wave_length: Optional[float] = None
        self._period: Optional[float] = None
        self._celerity: Optional[float] = None
        # Processing attributes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_image: Optional[WavesImage] = None
        self.radon_transform: Optional[WavesRadon] = None
        # Filters
        self.correlation_image_filters: ImageProcessingFilters = [(detrend, []), (
            clipping, [self._parameters.TUNING.RATIO_SIZE_CORRELATION])]
        self.radon_image_filters: SignalProcessingFilters = [
            (remove_median, [self._parameters.TUNING.MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM]),
            (filter_mean, [self._parameters.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM])]
        # Intern attributes
        self._angles: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        self._variance: Optional[np.ndarray] = None
        self._sinogram_max_var: Optional[WavesSinogram] = None
        self._temporal_signal: Optional[np.ndarray] = None
        self._temporal_peaks_max: Optional[np.ndarray] = None
        self._dephasing: Optional[float] = None
        self._signal_period: Optional[np.ndarray] = None
        self._duration: Optional[float] = None
        self._temporal_arg_peaks_max: Optional[np.ndarray] = None
        self._wave_length_zeros: Optional[np.ndarray] = None

    def create_waves_field_estimation(self, direction: float, wavelength: float
                                      ) -> CorrelationWavesFieldEstimation:
        """ Creates the WavesFieldEstimation instance where the local estimator will store its
        estimations.

        :param direction: the propagation direction of the waves field (degrees measured clockwise
                          from the North).
        :param wavelength: the wavelength of the waves field
        :returns: an initialized instance of WavesFilesEstimation to be filled in further on.
        """
        waves_field_estimation = CorrelationWavesFieldEstimation(self.gravity,
                                                                 self.local_estimator_params.DEPTH_EST_METHOD,
                                                                 self.local_estimator_params.D_PRECISION)
        waves_field_estimation.direction = direction
        waves_field_estimation.wavelength = wavelength

        return waves_field_estimation

    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        try:
            self.correlation_image.apply_filters(self.correlation_image_filters)
            self.radon_transform = WavesRadon(self.correlation_image)
            # It is very important that circle=True has been chosen to compute radon matrix since
            # we read values in meters from the axis of the sinogram
            self.radon_transform.compute()
            self.radon_transform.apply_filter(self.radon_image_filters)
            self._sinogram_max_var, self._direction_propagation, self._variance = \
                self.radon_transform.get_sinogram_maximum_variance()
            self.compute_wave_length()
            self.compute_celerity()
            self.temporal_reconstruction()
            self.temporal_reconstruction_tuning()
            self.compute_period()
            waves_field_estimation = self.create_waves_field_estimation(self.direction_propagation,
                                                                        self.wave_length)
            waves_field_estimation.period = self.period
            waves_field_estimation.celerity = self.celerity
            self.store_estimation(waves_field_estimation)
        except Exception as excp:
            print(f'Bathymetry computation failed: {str(excp)}')

    @property
    @abstractmethod
    def _parameters(self) -> Munch:
        """ :return: munchified parameters
        """
        # FIXME: Why not using parameters from global bathy estimatror (this is
        # the general principle)

    @property
    @abstractmethod
    def sampling_positions(self) -> np.ndarray:
        """ :return: ndarray of x positions
        """

    @abstractmethod
    def get_correlation_matrix(self) -> np.ndarray:
        """ :return: correlation matrix
        """

    def get_correlation_image(self) -> WavesImage:
        """ :return: correlation image
        """
        return WavesImage(self.correlation_matrix, self._parameters.RESOLUTION.SPATIAL)

    @property
    def preprocessing_filters(self) -> ImageProcessingFilters:
        """ :returns: A list of functions together with their parameters to be applied
        sequentially to all the images of the sequence before subsequent bathymetry estimation.
        """
        preprocessing_filters: ImageProcessingFilters = []
        return preprocessing_filters

    @property
    def direction_propagation(self) -> float:
        if self._direction_propagation is None:
            raise AttributeError('No direction propagation computed yet')
        return self._direction_propagation

    @property
    def celerity(self) -> float:
        if self._celerity is None:
            raise AttributeError('No celerity computed yet')
        return self._celerity

    @property
    def period(self) -> float:
        if self._period is None:
            raise AttributeError('No period computed yet')
        return self._period

    @property
    def wave_length(self) -> float:
        if self._wave_length is None:
            raise AttributeError('No wave length computed yet')
        return self._wave_length

    def get_angles(self) -> np.ndarray:
        """ Get the angles between all points selected to compute correlation

        :return: Angles (in degrees)
        """
        xrawipool_ik_dist = \
            np.tile(self.sampling_positions[0], (len(self.sampling_positions[0]), 1)) - \
            np.tile(self.sampling_positions[0].T, (1, len(self.sampling_positions[0])))
        yrawipool_ik_dist = \
            np.tile(self.sampling_positions[1], (len(self.sampling_positions[1]), 1)) - \
            np.tile(self.sampling_positions[1].T, (1, len(self.sampling_positions[1])))
        return np.arctan2(xrawipool_ik_dist, yrawipool_ik_dist).T * 180 / np.pi

    def get_distances(self) -> np.ndarray:
        """ Distances between positions x and positions y
        Be aware these distances are not in meter and have to be multiplied by spatial resolution

        :return: the distances between all points selected to compute correlation
        """
        return np.sqrt(
            np.square((self.sampling_positions[0] - self.sampling_positions[0].T)) +
            np.square((self.sampling_positions[1] - self.sampling_positions[1].T)))

    @property
    def correlation_image(self) -> WavesImage:
        """ :return: correlation image used to perform radon transformation
        """
        if self._correlation_image is None:
            self._correlation_image = self.get_correlation_image()
        return self._correlation_image

    @property
    def sinogram_max_var(self) -> WavesSinogram:
        if self._sinogram_max_var is None:
            raise AttributeError('No sinogram computed yet')
        return self._sinogram_max_var

    @property
    def correlation_matrix(self) -> np.ndarray:
        """ Be aware this matrix is projected before radon transformation in temporal correlation
        case

        :return: correlation matrix used for temporal reconstruction
        """
        if self._correlation_matrix is None:
            self._correlation_matrix = self.get_correlation_matrix()
        return self._correlation_matrix

    @property
    def angles(self) -> np.ndarray:
        """ :return: angles in radian
        """
        if self._angles is None:
            self._angles = self.get_angles()
        return self._angles

    @property
    def distances(self) -> np.ndarray:
        """ :return: distances
        """
        if self._distances is None:
            self._distances = self.get_distances()
        return self._distances

    def compute_wave_length(self) -> None:
        """ Wave length computation (in meter)
        """
        period, self._wave_length_zeros = find_period(self.sinogram_max_var.sinogram.flatten())
        self._wave_length = period * self._parameters.RESOLUTION.SPATIAL

    def compute_celerity(self) -> None:
        """ Celerity computation (in meter/second)
        """
        self._dephasing, self._signal_period = find_dephasing(self.sinogram_max_var.sinogram,
                                                              self.wave_length)
        rhomx = self._parameters.RESOLUTION.SPATIAL * self._dephasing
        self._duration = self.global_estimator.get_delta_time(
            self._position)
        self._celerity = np.abs(rhomx / self._duration)

    def temporal_reconstruction(self) -> None:
        """ Temporal reconstruction of the correlation signal following propagation direction
        """
        distances = np.cos(np.radians(self.direction_propagation - self.angles.T.flatten())) * \
            self.distances.flatten() * self._parameters.RESOLUTION.SPATIAL
        time = distances / self.celerity
        time_unique, index_unique = np.unique(time, return_index=True)
        index_unique_sorted = np.argsort(time_unique)
        time_unique_sorted = time_unique[index_unique_sorted]
        timevec = np.arange(np.min(time_unique_sorted), np.max(time_unique_sorted),
                            self._parameters.RESOLUTION.TIME_INTERPOLATION)
        corr_unique_sorted = self.correlation_matrix.T.flatten()[
            index_unique[index_unique_sorted]]
        interpolation = interp1d(time_unique_sorted, corr_unique_sorted)
        self._temporal_signal = interpolation(timevec)

    def temporal_reconstruction_tuning(self) -> None:
        """ Tuning of temporal signal
        """
        low_frequency = self._parameters.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION * \
            self._parameters.RESOLUTION.TIME_INTERPOLATION
        high_frequency = self._parameters.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION * \
            self._parameters.RESOLUTION.TIME_INTERPOLATION
        sos_filter = butter(1, (2 * low_frequency, 2 * high_frequency),
                            btype='bandpass', output='sos')
        self._temporal_signal = sosfiltfilt(sos_filter, self._temporal_signal)

    def compute_period(self) -> None:
        """Period computation (in second)
        """
        self._temporal_arg_peaks_max, _ = find_peaks(
            self._temporal_signal, distance=self._parameters.TUNING.MIN_PEAKS_DISTANCE_PERIOD)
        self._period = float(np.mean(np.diff(self._temporal_arg_peaks_max)))
