# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using spatial correlation method

:author: Grégoire Thoumyre
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 20/09/2021
"""
from typing import Optional, List, TYPE_CHECKING, cast  # @NoMove

from scipy.signal import find_peaks
from shapely.geometry import Point

import numpy as np

from ..bathy_physics import celerity_offshore, wavelength_offshore, period_offshore
from ..generic_utils.image_filters import detrend, desmooth
from ..generic_utils.image_utils import normalized_cross_correlation
from ..generic_utils.signal_utils import find_period_from_zeros
from ..image.ortho_sequence import OrthoSequence, FrameIdType
from ..image_processing.sinograms import Sinograms
from ..image_processing.waves_image import ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, linear_directions
from ..image_processing.waves_sinogram import WavesSinogram
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator
from .spatial_correlation_bathy_estimation import SpatialCorrelationBathyEstimation


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class SpatialCorrelationBathyEstimator(LocalBathyEstimator):
    """ Class performing spatial correlation to compute bathymetry
    """

    wave_field_estimation_cls = SpatialCorrelationBathyEstimation

    def __init__(self, location: Point, ortho_sequence: OrthoSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(location, ortho_sequence, global_estimator, selected_directions)

        if self.selected_directions is None:
            self.selected_directions = linear_directions(-180., 0., 1.)

        self.radon_transforms: List[Sinograms] = []
        self.sinograms: List[WavesSinogram] = []
        self.spatial_correlation = None
        self.directions = None

    @property
    def start_frame_id(self) -> FrameIdType:
        return self.global_estimator.selected_frames[0]

    @property
    def stop_frame_id(self) -> FrameIdType:
        return self.global_estimator.selected_frames[1]

    @property
    def radon_augmentation_factor(self) -> float:
        """ The factor by which the spatial resolution must be divided in order to improve the
        accuracy of the propagation distance estimation.
        """
        return self.local_estimator_params['AUGMENTED_RADON_FACTOR']

    @property
    def augmented_resolution(self) -> float:
        """ The augmented spatial resolution at which the propagation distance estimation is done.
        """
        return self.spatial_resolution * self.radon_augmentation_factor

    @property
    def preprocessing_filters(self) -> ImageProcessingFilters:
        preprocessing_filters: ImageProcessingFilters = []
        preprocessing_filters.append((detrend, []))

        if self.global_estimator.smoothing_requested:
            # FIXME: pixels necessary for smoothing are not taken into account, thus
            # zeros are introduced at the borders of the window.
            preprocessing_filters.append((desmooth,
                                          [self.global_estimator.smoothing_lines_size,
                                           self.global_estimator.smoothing_columns_size]))
            # Remove tendency possibly introduced by smoothing, specially on the shore line
            preprocessing_filters.append((detrend, []))
        return preprocessing_filters

    def run(self) -> None:
        self.preprocess_images()  # TODO: should be in the init ?
        estimated_direction = self.find_direction()
        self.compute_radon_transforms(estimated_direction)
        correlation_signal = self.compute_spatial_correlation(estimated_direction)
        wavelength = self.compute_wavelength(correlation_signal)
        delta_position = self.compute_delta_position(correlation_signal, wavelength)
        self.save_wave_field_estimation(estimated_direction, wavelength, delta_position)

    def compute_radon_transforms(self, estimated_direction: float) -> None:
        """ Compute the augmented Radon transforms of all the images in the sequence along a single
        direction.
        """
        for image in self.ortho_sequence:
            radon_transform = WavesRadon(image, np.array([estimated_direction]))
            radon_transform_augmented = radon_transform.radon_augmentation(
                self.radon_augmentation_factor)
            self.radon_transforms.append(radon_transform_augmented)

    def find_direction(self) -> float:
        """ Find the direction of the waves propagation using the radon transform of the first
        image in the sequence.

        :returns: the estimated direction of the waves propagation
        """
        tmp_wavesradon = WavesRadon(self.ortho_sequence[0], self.selected_directions)
        estimated_direction, _ = tmp_wavesradon.get_direction_maximum_variance()
        return estimated_direction

    def compute_spatial_correlation(self, estimated_direction: float) -> np.ndarray:
        """ Compute the spatial cross correlation between the 2 sinograms of the estimated direction

        :param estimated_direction: the estimated direction of the waves propagation
        :returns: the correlation signal
        """
        for radon_transform in self.radon_transforms:
            tmp_wavessinogram = radon_transform[estimated_direction]
            tmp_wavessinogram.values *= tmp_wavessinogram.variance
            self.sinograms.append(tmp_wavessinogram)
        sinogram_1 = self.sinograms[0].values
        # TODO: should be independent from 0/1 (for multiple pairs of frames)
        sinogram_2 = self.sinograms[1].values
        correl_mode = self.local_estimator_params['CORRELATION_MODE']
        corr_init = normalized_cross_correlation(sinogram_1, sinogram_2, correl_mode)
        corr_init_ac = normalized_cross_correlation(corr_init, corr_init, correl_mode)
        corr_1 = normalized_cross_correlation(corr_init_ac, sinogram_1, correl_mode)
        corr_2 = normalized_cross_correlation(corr_init_ac, sinogram_2, correl_mode)
        correlation_signal = normalized_cross_correlation(corr_1, corr_2, correl_mode)
        return correlation_signal

    def compute_wavelength(self, correlation_signal: np.ndarray) -> float:
        """ Compute the wave length of the waves

        :param correlation_signal: spatial cross correlated signal
        :returns: the wave length (m)
        """
        min_wavelength = wavelength_offshore(self.global_estimator.waves_period_min, self.gravity)
        min_period_unitless = int(min_wavelength / self.augmented_resolution)
        period, _ = find_period_from_zeros(correlation_signal, min_period_unitless)
        wavelength = period * self.augmented_resolution
        return wavelength

    def compute_delta_position(self, correlation_signal: np.ndarray,
                               wavelength: float) -> float:
        """ Compute the distance propagated over time by the waves

        :param correlation_signal: spatial cross correlated signal
        :param wavelength: the wave length (m)
        :returns: the distance propagated over time by the waves (m)
        :raises WavesEstimationError: when no directional peak can be found
        """
        peaks_pos, _ = find_peaks(correlation_signal)
        if peaks_pos.size == 0:
            raise WavesEstimationError('Unable to find any directional peak')
        argmax_ac = len(correlation_signal) // 2
        relative_distance = (peaks_pos - argmax_ac) * self.augmented_resolution

        celerity_offshore_max = celerity_offshore(self.global_estimator.waves_period_max,
                                                  self.gravity)
        spatial_shift_offshore_max = celerity_offshore_max * self.propagation_duration
        spatial_shift_min = min(-spatial_shift_offshore_max, spatial_shift_offshore_max)
        spatial_shift_max = -spatial_shift_min

        stroboscopic_factor_offshore = self.propagation_duration / period_offshore(1. / wavelength,
                                                                                   self.gravity)
        if abs(stroboscopic_factor_offshore) >= 1:
            # unused for s2
            print('test stroboscopie vrai')
            spatial_shift_offshore_max = self.local_estimator_params['PEAK_POSITION_MAX_FACTOR'] \
                * stroboscopic_factor_offshore * wavelength

        pt_in_range = peaks_pos[np.where((relative_distance >= spatial_shift_min) &
                                         (relative_distance < spatial_shift_max))]
        if pt_in_range.size == 0:
            raise WavesEstimationError('Unable to find any directional peak')
        argmax = pt_in_range[correlation_signal[pt_in_range].argmax()]
        delta_position = (argmax_ac - argmax) * self.augmented_resolution

        return delta_position

    def save_wave_field_estimation(self,
                                   estimated_direction: float,
                                   wavelength: float,
                                   delta_position: float) -> None:
        """ Saves the wave_field_estimation

        :param estimated_direction: the waves estimated propagation direction
        :param wavelength: the wave length of the waves
        :param delta_position: the distance propagated over time by the waves
        """
        bathymetry_estimation = cast(SpatialCorrelationBathyEstimation,
                                     self.create_bathymetry_estimation(estimated_direction,
                                                                       wavelength))
        bathymetry_estimation.delta_position = delta_position
        self.bathymetry_estimations.append(bathymetry_estimation)
