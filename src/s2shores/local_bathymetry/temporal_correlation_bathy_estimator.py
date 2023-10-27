# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import Optional, Tuple, TYPE_CHECKING, cast  # @NoMove

import numpy as np
import pandas
from scipy.signal import find_peaks
from shapely.geometry import Point

from ..bathy_physics import wavelength_offshore
from ..generic_utils.image_filters import detrend, clipping, normalise, gaussian_masking
from ..generic_utils.image_utils import cross_correlation
from ..generic_utils.signal_filters import detrend_signal, butter_bandpass_filter, filter_median
from ..generic_utils.signal_utils import find_period_from_zeros
from ..image.ortho_sequence import OrthoSequence, FrameIdType
from ..image_processing.waves_image import WavesImage, ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon, linear_directions
from ..image_processing.waves_sinogram import WavesSinogram
from ..waves_exceptions import WavesEstimationError, NotExploitableSinogram, CorrelationComputationError, SequenceImagesError

from .local_bathy_estimator import LocalBathyEstimator
from .temporal_correlation_bathy_estimation import TemporalCorrelationBathyEstimation

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class TemporalCorrelationBathyEstimator(LocalBathyEstimator):
    """ Class performing temporal correlation to compute bathymetry
    """

    wave_field_estimation_cls = TemporalCorrelationBathyEstimation

    def __init__(self, location: Point, ortho_sequence: OrthoSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(location, ortho_sequence, global_estimator, selected_directions)

        if self.selected_directions is None:
            self.selected_directions = linear_directions(-90., 90., 1.)
        # Processing attributes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_image: Optional[WavesImage] = None
        self.radon_transform: Optional[WavesRadon] = None
        self.sinogram_maxvar: Optional[WavesSinogram] = None
        self._angles: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        self._sampling_positions: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._time_series: Optional[np.ndarray] = None
        self._sampling_period: Optional[float] = None

        # Correlation filters 
        self.correlation_image_filters: ImageProcessingFilters = [(detrend, []),
                                                                  (gaussian_masking, [self.local_estimator_params['TUNING']['SIGMA_CORRELATION_MASK']]),
                                                                  (clipping, [self.local_estimator_params['TUNING']['RATIO_SIZE_CORRELATION']])]
        # Projected sinogram filter
        self.sinogram_max_var_filters: SignalProcessingFilters = [(filter_median,[self.local_estimator_params['TUNING']['MEDIAN_FILTER_KERNEL']])]
        
        # Check if time lag is valid
        if self.local_estimator_params['TEMPORAL_LAG'] >= len(self.ortho_sequence):
            raise ValueError(
                'The chosen number of lag frames is bigger than the number of available frames')
            
        if self.debug_sample:
            self.metrics['sampling_duration'] = self.sampling_period
            self.metrics['propagation_duration'] = self.propagation_duration
            self.metrics['spatial_resolution'] = self.spatial_resolution        


    @property
    def start_frame_id(self) -> FrameIdType:
        return int(self.global_estimator.selected_frames[0])

    @property
    def stop_frame_id(self) -> FrameIdType:
        return cast(int, self.start_frame_id) + self.nb_lags

    @property
    def nb_lags(self) -> int:
        """ :returns: the number of lags (interval between 2 frames) to use
        """
        return self.local_estimator_params['TEMPORAL_LAG']
    
    @property
    def sampling_period(self) -> float:
        """ Sampling period between each frame
        :return: Image sequence sampling period
        """
        if self._sampling_period is None:
            self._sampling_period = self.ortho_sequence.get_time_difference(self._location, 1, 2)
        return self._sampling_period
    
    @property
    def preprocessing_filters(self) -> ImageProcessingFilters:
        """ :returns: A list of functions together with their parameters to be applied
        sequentially to all the images of the sequence before subsequent bathymetry estimation.
        """
        preprocessing_filters: ImageProcessingFilters = []
        preprocessing_filters.append((normalise, []))
        return preprocessing_filters
    
    @property
    def sampling_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """ :returns: tuple of sampling positions
        :raises ValueError: when sampling has not been defined
        """
        if self._sampling_positions is None:
            raise ValueError('Sampling positions are not defined')
        return self._sampling_positions
    
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
        :raises CorrelationComputationError:  when correlation matrix can not be computed
        :raises SequenceImagesError: when the time series is not defined
        """
        if self._correlation_matrix is None:
            if self._time_series is None:
                raise SequenceImagesError('Time series are not defined')
            try:
                self._correlation_matrix = cross_correlation(self._time_series[:, :-self.nb_lags],
                                                             self._time_series[:, self.nb_lags:])
            except ValueError as excp:
                raise CorrelationComputationError(
                    'Cross correlation can not be computed because of standard deviation of 0') from excp
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
    
    @property
    def adaptative_window_limit(self) -> float:
        """ :returns: Maximal wavelength limitation to enable adaptative window
        """
        return np.min(self.ortho_sequence.shape)/self.local_estimator_params['TUNING']['ADAPTATIVE_RATIO']*self.ortho_sequence.resolution
         
    
    def run(self) -> None:
        """ Run the local bathy estimator using correlation method
        """
        
        # Normalise each frame
        self.preprocess_images()
        
        # Select random pixel position within the frame stack and extract Time-series 
        self.create_sequence_time_series()

        # Compute correlation and apply a gaussian mask
        self.compute_temporal_correlation()

        # Radon transform (sinogram) and get wave direction
        direction_propagation = self.compute_radon_transform()
        
        # Extract wavelength and delta_x
        wavelength, distances = self.compute_wavefield()

        if wavelength<self.adaptative_window_limit:
            
            # Adapt window size with estimated wavelength
            self.adapt_window(wavelength)
            
            # Select random pixel position within the frame stack and extract Time-series 
            self.create_sequence_time_series()

            # Compute correlation and apply a gaussian mask
            self.compute_temporal_correlation()

            # Radon transform (sinogram) and get wave direction
            direction_propagation = self.compute_radon_transform()

            # Extract wavelength and delta_x
            wavelength, distances = self.compute_wavefield()

        # Save the estimation
        self.save_wave_field_estimation(direction_propagation, wavelength, distances)
            
            
    def adapt_window(self, wavelength: float) -> None:
        """ Crop the ortho_sequence according to the wavelength adapted window. 
            Reset the estimator variable used in the depth extraction.
            
        :param wavelength: wavelength estimated in the first iteration     
        """
        # Processing attributes
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_image: Optional[WavesImage] = None
        self.radon_transform: Optional[WavesRadon] = None
        self.sinogram_maxvar: Optional[WavesSinogram] = None
        self._angles: Optional[np.ndarray] = None
        self._distances: Optional[np.ndarray] = None
        self._sampling_positions: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._time_series: Optional[np.ndarray] = None
        self._sampling_period: Optional[float] = None
        
        image_size = self.ortho_sequence.shape
        adapted_window = np.floor(wavelength*self.local_estimator_params['TUNING']['ADAPTATIVE_RATIO']/self.ortho_sequence.resolution)
        
        if adapted_window>=np.min(image_size):
            return
        window_size = (int(image_size[0]//2 - adapted_window//2), int(image_size[0]//2 + adapted_window//2),
                       int(image_size[1]//2 - adapted_window//2), int(image_size[1]//2 + adapted_window//2))

        for image in self.ortho_sequence:
            filtered_image = image.extract_sub_image(window_size)
            image.pixels = filtered_image.pixels
            
            
    def create_sequence_time_series(self) -> None:
        """ This function computes an np.array of time series.
        To do this random points are selected within the sequence of image and a temporal series
        is included in the np.array for each selected point
        """
        percentage_points = self.local_estimator_params['PERCENTAGE_POINTS']
        if percentage_points < 0 or percentage_points > 100:
            raise ValueError('Percentage must be between 0 and 100')

        # Create frame stack
        merge_array = np.dstack([image.pixels for image in self.ortho_sequence])
        merge_array[np.isnan(merge_array)] = 0 # Set nan to 0
        
        # Select pixel positions randomly
        shape_y, shape_x = self.ortho_sequence.shape
        image_size = shape_x * shape_y
        time_series = np.reshape(merge_array, (image_size, -1))
        np.random.seed(0) # A seed is used here to reproduce same results
        nb_random_points = round(image_size * percentage_points / 100)
        random_indexes = np.random.randint(image_size, size=nb_random_points)

        sampling_positions_x, sampling_positions_y = np.unravel_index(random_indexes, self.ortho_sequence.shape)
        self._sampling_positions = (np.reshape(sampling_positions_x, (1, -1)),
                                    np.reshape(sampling_positions_y, (1, -1)))

        # Extract and detrend Time-series
        time_series_detrend = detrend_signal(time_series[random_indexes, :], axis=1)
        
        # BP filtering, bypassed if low or high cutoff is set to 0
        if self.local_estimator_params['TUNING']['LOWCUT_PERIOD']!=0 and self.local_estimator_params['TUNING']['HIGHCUT_PERIOD']!=0:
            fps = 1/self.sampling_period
            self._time_series = butter_bandpass_filter(time_series_detrend, 
                                                       lowcut_period=self.local_estimator_params['TUNING']['LOWCUT_PERIOD'], 
                                                       highcut_period=self.local_estimator_params['TUNING']['HIGHCUT_PERIOD'], 
                                                       fs=fps, 
                                                       axis=1)
        else:
            self._time_series = time_series_detrend                

        if self.debug_sample:
            self.metrics['detrend_time_series'] = time_series_detrend[0,:]
            self.metrics['filtered_time_series'] = self._time_series[0,:]

    
    def compute_temporal_correlation(self) -> None:
        """ Compute the temporal correlation matrix using the created time-series sequence
        """
        filtered_correlation = self.correlation_image.apply_filters(self.correlation_image_filters)
        self.correlation_image.pixels = filtered_correlation.pixels
    
    
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

        projected_matrix = np.nanmean(self.correlation_matrix) * np.ones(
            (np.max(indices_x) + 1, np.max(indices_y) + 1))
        projected_matrix[indices_x, indices_y] = values
        
        if self.debug_sample:
            self.metrics['corr_indices_x'] = indices_x
            self.metrics['corr_indices_y'] = indices_y
            self.metrics['projected_corr_raw'] = projected_matrix

        return WavesImage(projected_matrix, self.spatial_resolution)

    
    def compute_radon_transform(self) -> float:
        """Compute the Rason Transform from the correlation matrix and determine wave progation direction
        
        :returns: wave propagation direction
        """
        self.radon_transform = WavesRadon(self.correlation_image, self.selected_directions)

        # Compute propagation angle using sinogram max variance
        direction_propagation, variances = self.radon_transform.get_direction_maximum_variance()
        
        # Extract projected sinogram at max var ang from sinogram
        self.sinogram_maxvar = self.radon_transform[direction_propagation]
        
        # Median filtering of the projected sinogram
        filtered_sinogram_maxvar = self.sinogram_maxvar.apply_filters(self.sinogram_max_var_filters)
        self.sinogram_maxvar.values = filtered_sinogram_maxvar.values
        
        if self.debug_sample:
            self.metrics['corr_radon_input'] = self.radon_transform.pixels
            self.metrics['radon_transform'] = self.radon_transform
            self.metrics['variances'] = variances
            self.metrics['direction'] = direction_propagation
            self.metrics['sinogram_max_var'] = self.sinogram_maxvar.values
            
        return direction_propagation
    
    
    def compute_wavefield(self) -> list:
        """Extract wave chracteristics (wavelength and wave displacement) within the projected sinogram
        
        :returns: estimated wavelength and wave displacement
        """
        # Extract wavelength from sinogram projected at max var angle (0-crossing)
        wavelength, wave_length_zeros = self.compute_wavelength()
        
        # Extract delta_x of the wave within time_lag from sinogram projected at max var angle (peaks)
        distance = self.compute_distance(wavelength, wave_length_zeros)
        
        return wavelength, distance
    
    
    def compute_wavelength(self) -> float:
        """ Wavelength computation (in meter)
        
        :param sinogram : sinogram used to compute wave length
        :returns: wavelength and 0-crossing positions
        :raises NotExploitableSinogram: if wave length can not be computed from sinogram
        """
        min_wavelength = wavelength_offshore(self.global_estimator.waves_period_min, self.gravity)
        try:
            period, wave_length_zeros = find_period_from_zeros(self.sinogram_maxvar.values, int(min_wavelength / self.spatial_resolution))
        except ValueError as excp:
            raise NotExploitableSinogram('Wave length can not be computed from sinogram') from excp
        wave_length = period * self.spatial_resolution

        if self.debug_sample:
            self.metrics['wave_spatial_zeros'] = wave_length_zeros * self.spatial_resolution
        return wave_length, wave_length_zeros

    
    def compute_distance(self, wavelength: float, zeros: float) -> np.ndarray:
        """ Propagated distance computation (in meter)

        :param wavelength: wavelength estimated in the projected sinogram 
        :param zeros: zeros-crossing positions
        :returns: propagation distances of the wave
        """
        sinogram = self.sinogram_maxvar.values
        x_axis = np.arange(0, len(sinogram))-(len(sinogram) // 2)
        period = int(wavelength / self.spatial_resolution)
        max_sinogram = np.max(sinogram[(x_axis >= zeros[0]) & (x_axis < zeros[-1])])
        
        # Find peaks
        tuning_parameters = self.local_estimator_params['TUNING']
        peaks, _ = find_peaks(sinogram[(x_axis >= zeros[0]) & (x_axis < zeros[-1])], 
                              height=tuning_parameters['PEAK_DETECTION_HEIGHT_RATIO']*max_sinogram,
                              distance=tuning_parameters['PEAK_DETECTION_DISTANCE_RATIO']*period)
        
        # Compute initial distance
        dx_in_list = x_axis[(x_axis >= zeros[0]) & (x_axis < zeros[-1])][peaks]
        distances = []
        
        for dx_in in dx_in_list:
            # Find 0-crossing surrounding the peak
            z1 = zeros[zeros>dx_in][0]
            z2 = zeros[zeros<=dx_in][-1]

            # Refine distance
            ref = [z1,z2][np.argmin(np.abs([z1,z2]))]
            offset = np.sign(dx_in-ref)*(np.abs(z2-z1)/2)
            dx = ref + offset
            distances.append(dx)
            
        # Distance of the wave propagation
        distances = np.array(distances) * self.spatial_resolution
        
        if self.debug_sample:
            self.metrics['max_indices'] = peaks
            self.metrics['wave_distance'] = distances
        
        return distances

    
    def save_wave_field_estimation(self, estimated_direction: float, wavelength: float, distances: list) -> None:
        """ Saves the wavefield estimations

        :param estimated_direction: the waves estimated propagation direction
        :param wavelength: the wave length of the waves
        :param distances: the wave distances extracted from the projected sinogram
        """

        for distance in distances:
            bathymetry_estimation = cast(TemporalCorrelationBathyEstimation, 
                                         self.create_bathymetry_estimation(estimated_direction, wavelength)
                                        )
            bathymetry_estimation.delta_position = distance
            
            self.bathymetry_estimations.append(bathymetry_estimation)
        self.bathymetry_estimations.sort_on_attribute('linearity', reverse=False)
        
        if self.debug_sample:
            self.metrics['bathymetry_estimation'] = self.bathymetry_estimations
            self.metrics['status'] = self.bathymetry_estimations.status
        
        