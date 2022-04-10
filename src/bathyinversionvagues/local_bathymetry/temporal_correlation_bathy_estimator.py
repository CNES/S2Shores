# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from copy import deepcopy
from typing import Optional, Tuple, TYPE_CHECKING  # @NoMove


import pandas
from scipy.signal import find_peaks
import numpy as np


from ..bathy_physics import wavelength_offshore
from ..generic_utils.image_filters import detrend, clipping
from ..generic_utils.image_utils import cross_correlation
from ..generic_utils.signal_filters import filter_mean, remove_median
from ..generic_utils.signal_utils import find_period_from_zeros
from ..image.image_geometry_types import PointType
from ..image_processing.images_sequence import ImagesSequence
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, linear_directions
from ..image_processing.waves_sinogram import SignalProcessingFilters
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator
from .temporal_correlation_bathy_estimation import TemporalCorrelationBathyEstimation


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class TemporalCorrelationBathyEstimator(LocalBathyEstimator):
    """ Class performing temporal correlation to compute bathymetry
    """
    wave_field_estimation_cls = TemporalCorrelationBathyEstimation

    def __init__(self, location: PointType, images_sequence: ImagesSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(location, images_sequence, global_estimator, selected_directions)

        if self.selected_directions is None:
            self.selected_directions = linear_directions(-180., 0., 1.)
        # Processing attributes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_image: Optional[WavesImage] = None
        self.radon_transform: Optional[WavesRadon] = None
        self._angles: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        self._sampling_positions: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._time_series: Optional[np.ndarray] = None

        # Filters
        self.correlation_image_filters: ImageProcessingFilters = [(detrend, []), (
            clipping, [self.local_estimator_params['TUNING']['RATIO_SIZE_CORRELATION']])]
        self.radon_image_filters: SignalProcessingFilters = [
            (remove_median,
             [self.local_estimator_params['TUNING']['MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM']]),
            (filter_mean,
             [self.local_estimator_params['TUNING']['MEAN_FILTER_KERNEL_SIZE_SINOGRAM']])]

    @property
    def nb_lag_frames(self) -> int:
        # FIXME: this is not what we want, but needed to retrieve same propagation_duration
        return self.local_estimator_params['TEMPORAL_LAG'] + 1

    def create_sequence_time_series(self) -> None:
        """ This function computes an np.array of time series.
        To do this random points are selected within the sequence of image and a temporal serie
        is included in the np.array for each selected point
        """
        percentage_points = self.local_estimator_params['PERCENTAGE_POINTS']
        if percentage_points < 0 or percentage_points > 100:
            raise ValueError('Percentage must be between 0 and 100')
        merge_array = np.dstack([image.pixels for image in self.images_sequence])
        shape_x, shape_y = self.images_sequence.shape
        image_size = shape_x * shape_y
        time_series = np.reshape(merge_array, (image_size, -1))
        # A seed is used here to reproduce same results
        np.random.seed(0)
        nb_random_points = round(image_size * percentage_points / 100)
        random_indexes = np.random.randint(0, image_size, size=nb_random_points)
        positions_y, positions_x = np.meshgrid(np.linspace(1, shape_x, shape_x),
                                               np.linspace(1, shape_y, shape_y))

        sampling_positions_x = np.reshape(positions_x.flatten()[random_indexes], (1, -1))
        sampling_positions_y = np.reshape(positions_y.flatten()[random_indexes], (1, -1))
        self._sampling_positions = (sampling_positions_x, sampling_positions_y)
        self._time_series = time_series[random_indexes, :]

    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        try:
            self.create_sequence_time_series()
            filtered_image = self.correlation_image.apply_filters(self.correlation_image_filters)
            self.correlation_image.pixels = filtered_image.pixels
            radon_transform = WavesRadon(self.correlation_image, self.selected_directions)
            filtered_radon = radon_transform.apply_filters(self.radon_image_filters)
            direction_propagation, variances = filtered_radon.get_direction_maximum_variance()
            sinogram_max_var = radon_transform[direction_propagation]
            sinogram_max_var_values = sinogram_max_var.values
            wavelength = self.compute_wavelength(sinogram_max_var_values)
            distances = self.compute_distances(sinogram_max_var_values, wavelength,
                                               nb_hops=self.local_estimator_params['HOPS_NUMBER'])

            # Keep in mind that direction_estimations stores several estimations for a same
            # direction and only the best of them should be added in the final list
            # direction_estimation is empty at this point
            direction_estimations = deepcopy(self.bathymetry_estimations)

            for distance in distances:
                estimation = self.create_bathymetry_estimation(direction_propagation,
                                                               wavelength)
                estimation.delta_position = distance
                direction_estimations.append(estimation)

            celerities = direction_estimations.get_attribute('celerity')
            linearity_coefficients = direction_estimations.get_attribute('linearity')
            direction_estimations.remove_unphysical_wave_fields()
            if not direction_estimations:
                raise WavesEstimationError('No correct wave fied estimations have been found')
            direction_estimations.sort_on_attribute('linearity', reverse=False)
            best_estimation = direction_estimations[0]

            self.bathymetry_estimations.append(best_estimation)

            if self.debug_sample:
                self.metrics['radon_transform'] = radon_transform
                self.metrics['variances'] = variances
                self.metrics['sinogram_max_var'] = sinogram_max_var_values
                # TODO: use objects in debug
                self.metrics['propagation_duration'] = self.propagation_duration
                self.metrics['distances'] = distances
                self.metrics['celerities'] = celerities
                self.metrics['linearity_coefficients'] = linearity_coefficients
        except Exception as excp:
            print(f'Bathymetry computation failed: {str(excp)}')

    @property
    def sampling_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ :returns: tuple of sampling positions
        :raises ValueError: when sampling has not been defined
        """
        if self._sampling_positions is None:
            raise ValueError('Sampling positions are not defined')
        return self._sampling_positions

    @property
    def preprocessing_filters(self) -> ImageProcessingFilters:
        """ :returns: A list of functions together with their parameters to be applied
        sequentially to all the images of the sequence before subsequent bathymetry estimation.
        """
        preprocessing_filters: ImageProcessingFilters = []
        return preprocessing_filters

    def get_correlation_image(self) -> WavesImage:
        """ This function computes the correlation image by projecting the correlation matrix
        on an array where axis are distances and center is the point where distance is 0.
        If several points have same coordinates, the mean of correlation is taken for this position
        """

        indices_x = np.round(self.distances * np.cos(self.angles))
        indices_x = np.array(indices_x - np.min(indices_x), dtype=int).T

        indices_y = np.round(self.distances * np.sin(self.angles))
        indices_y = np.array(indices_y - np.min(indices_y), dtype=int).T

        xr_s = pandas.Series(indices_x.flatten())
        yr_s = pandas.Series(indices_y.flatten())
        values_s = pandas.Series(self.correlation_matrix.flatten())

        # if two correlation values have same xr and yr mean of these values is taken
        dataframe = pandas.DataFrame({'xr': xr_s, 'yr': yr_s, 'values': values_s})
        dataframe_grouped = dataframe.groupby(by=['xr', 'yr']).mean().reset_index()
        values = np.array(dataframe_grouped['values'])
        indices_x = np.array(dataframe_grouped['xr'])
        indices_y = np.array(dataframe_grouped['yr'])

        projected_matrix = np.nanmean(self.correlation_matrix) * np.ones((np.max(indices_x) + 1,
                                                                          np.max(indices_y) + 1))
        projected_matrix[indices_x, indices_y] = values
        return WavesImage(projected_matrix, self.spatial_resolution)

    @property
    def correlation_image(self) -> WavesImage:
        """ Correlation image
        :return: correlation image used to perform radon transformation
        """
        if self._correlation_image is None:
            self._correlation_image = self.get_correlation_image()
        return self._correlation_image

    @property
    def correlation_matrix(self) -> np.ndarray:
        """ Compute temporal correlation matrix. Be aware this matrix is projected before radon
        transformation

        :return: correlation matrix used for temporal reconstruction
        """
        if self._correlation_matrix is None:
            if self._time_series is None:
                raise ValueError('Time series are not defined')
            # FIXME: 1 subtracted to compensate nb_lag_frames increment, but incorrect
            self._correlation_matrix = cross_correlation(
                self._time_series[:, self.nb_lag_frames - 1:],
                self._time_series[:, :-self.nb_lag_frames + 1])

        return self._correlation_matrix

    @property
    def angles(self) -> np.ndarray:
        """ :return: the angles between all points selected to compute correlation (in radians)
        """
        if self._angles is None:
            xrawipool_ik_dist = \
                np.tile(self.sampling_positions[0], (len(self.sampling_positions[0]), 1)) - \
                np.tile(self.sampling_positions[0].T, (1, len(self.sampling_positions[0])))
            yrawipool_ik_dist = \
                np.tile(self.sampling_positions[1], (len(self.sampling_positions[1]), 1)) - \
                np.tile(self.sampling_positions[1].T, (1, len(self.sampling_positions[1])))
            self._angles = np.arctan2(yrawipool_ik_dist, xrawipool_ik_dist)
        return self._angles

    @property
    def distances(self) -> np.ndarray:
        """ :return: Distances between all points selected to compute correlation. Be aware that
                     these distances are in pixels and must multiplied by spatial resolution
        """
        if self._distances is None:
            self._distances = np.sqrt(
                np.square((self.sampling_positions[0] - self.sampling_positions[0].T)) +
                np.square((self.sampling_positions[1] - self.sampling_positions[1].T)))
        return self._distances

    def compute_wavelength(self, sinogram: np.ndarray) -> float:
        """ Wavelength computation (in meter)
        :param sinogram : sinogram used to compute wave length
        :returns: wave length
        """
        min_wavelength = wavelength_offshore(self.global_estimator.waves_period_min, self.gravity)
        period, wave_length_zeros = find_period_from_zeros(
            sinogram, int(min_wavelength / self.spatial_resolution))
        wave_length = period * self.spatial_resolution

        if self.debug_sample:
            self.metrics['wave_length_zeros'] = wave_length_zeros
        return wave_length

    def compute_distances(self, sinogram: np.ndarray, wave_length: float,
                          nb_hops: int) -> np.ndarray:
        """ Propagated distance computation (in meter)
        Maxima are computed using peaks detection and the smallest nb_hops distances are selected

        :param sinogram: sinogram having maximum variance
        :param wave_length: wave_length computed on sinogram
        :param nb_hops : number of peaks to consider to compute distances
        :returns: np.ndarray of size nb_hops containing computed distances
        """
        x_axis = np.arange(-(len(sinogram) // 2), len(sinogram) // 2 + 1)
        interval = np.logical_and(x_axis * self.spatial_resolution > -wave_length,
                                  x_axis * self.spatial_resolution < wave_length)
        period = int(wave_length / self.spatial_resolution)
        max_sinogram = np.max(sinogram)
        tuning_parameters = self.local_estimator_params['TUNING']
        peaks, _ = find_peaks(sinogram, height=tuning_parameters['PEAK_DETECTION_HEIGHT_RATIO']
                              * max_sinogram,
                              distance=tuning_parameters['PEAK_DETECTION_DISTANCE_RATIO']
                              * period)
        distances = x_axis[peaks] * self.spatial_resolution
        index = np.argsort(sinogram[peaks])[::-1]
        if self.debug_sample:
            self.metrics['interval'] = interval
            self.metrics['x_axis'] = x_axis
            self.metrics['max_indices'] = peaks[index][:nb_hops]
        return distances[index][:nb_hops]
