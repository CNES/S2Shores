# -*- coding: utf-8 -*-
""" Class managing the computation of wave fields from two images taken at a small time interval.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING, cast  # @NoMove

from scipy.signal import find_peaks
from shapely.geometry import Point

import numpy as np

from ..bathy_physics import wavenumber_offshore
from ..generic_utils.image_filters import detrend, desmooth
from ..image.ortho_sequence import OrthoSequence, FrameIdType
from ..image_processing.waves_image import ImageProcessingFilters
from ..image_processing.waves_radon import WavesRadon
from ..waves_exceptions import WavesEstimationError

from .local_bathy_estimator import LocalBathyEstimator
from .spatial_dft_bathy_estimation import SpatialDFTBathyEstimation


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class SpatialDFTBathyEstimator(LocalBathyEstimator):
    """ A local bathymetry estimator estimating bathymetry from the DFT of the sinograms in
    radon transforms.
    """

    final_estimations_sorting = 'energy'
    wave_field_estimation_cls = SpatialDFTBathyEstimation

    def __init__(self, location: Point, ortho_sequence: OrthoSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:

        super().__init__(location, ortho_sequence, global_estimator, selected_directions)

        self.radon_transforms: List[WavesRadon] = []

        self.full_linear_wavenumbers = self.get_full_linear_wavenumbers()

    @property
    def start_frame_id(self) -> FrameIdType:
        return self.global_estimator.selected_frames[0]

    @property
    def stop_frame_id(self) -> FrameIdType:
        return self.global_estimator.selected_frames[1]

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

    def compute_radon_transforms(self) -> None:
        """ Compute the Radon transforms of all the images in the sequence using the currently
        selected directions.
        """
        for image in self.ortho_sequence:
            radon_transform = WavesRadon(image, self.selected_directions)
            self.radon_transforms.append(radon_transform)

    def run(self) -> None:
        """ Radon, FFT, find directional peaks, then do detailed DFT analysis to find
        detailed phase shifts per linear wave number (k*2pi)

        """
        self.preprocess_images()

        self.compute_radon_transforms()

        peaks_dir_indices = self.find_directions()

        directions_ranges = self.prepare_refinement(peaks_dir_indices)

        self.find_spectral_peaks(directions_ranges)

    def find_directions(self) -> np.ndarray:
        """ Find an initial set of directions from the cross correlation spectrum of the radon
        transforms of the 2 images.
        """
        # TODO: modify directions finding such that only one radon transform is computed (50% gain)
        sino1_fft = self.radon_transforms[0].get_sinograms_standard_dfts()
        sino2_fft = self.radon_transforms[1].get_sinograms_standard_dfts()

        phase_shift, spectrum_amplitude, sinograms_correlation_fft = \
            self._cross_correl_spectrum(sino1_fft, sino2_fft)
        total_spectrum = np.abs(phase_shift) * spectrum_amplitude

        max_heta = np.max(total_spectrum, axis=0)
        total_spectrum_normalized = max_heta / np.max(max_heta)

        # TODO: possibly apply symmetry to totalSpecMax_ref in find directions
        peaks, values = find_peaks(total_spectrum_normalized,
                                   prominence=self.local_estimator_params['PROMINENCE_MAX_PEAK'])
        prominences = values['prominences']

        # TODO: use symmetric peaks removal method (uncomment and delete next line.
        peaks = self._process_peaks(peaks, prominences)
        if peaks.size == 0:
            raise WavesEstimationError('Unable to find any directional peak')

        metrics: Dict[str, Any] = {}
        if self.debug_sample:
            metrics['sinograms_correlation_fft'] = sinograms_correlation_fft
            metrics['total_spectrum'] = total_spectrum
            metrics['max_heta'] = max_heta
            metrics['total_spectrum_normalized'] = total_spectrum_normalized
            self.metrics['standard_dft'] = metrics

        return peaks

    def _process_peaks(self, peaks: np.ndarray, prominences: np.ndarray) -> np.ndarray:
        # Find pairs of symmetric directions
        if self.debug_sample:
            print('initial peaks: ', peaks)
        peaks_pairs = []
        for index1 in range(peaks.size - 1):
            for index2 in range(index1 + 1, peaks.size):
                if abs(peaks[index1] - peaks[index2]) == 180:
                    peaks_pairs.append((index1, index2))
                    break
        if self.debug_sample:
            print('peaks_pairs: ', peaks_pairs)

        filtered_peaks_dir = []
        # Keep only one direction from each pair, with the greatest prominence
        for index1, index2 in peaks_pairs:
            if abs(prominences[index1] - prominences[index2]) < 100:
                # Prominences almost the same, keep lowest index
                filtered_peaks_dir.append(peaks[index1])
            else:
                if prominences[index1] > prominences[index2]:
                    filtered_peaks_dir.append(peaks[index1])
                else:
                    filtered_peaks_dir.append(peaks[index2])
        if self.debug_sample:
            print('peaks kept from peaks_pairs: ', filtered_peaks_dir)

        # Add peaks which do not belong to a pair
        for index in range(peaks.size):
            found_in_pair = False
            for index1, index2 in peaks_pairs:
                if index in (index1, index2):
                    found_in_pair = True
                    break
            if not found_in_pair:
                filtered_peaks_dir.append(peaks[index])
        if self.debug_sample:
            print('final peaks after adding isolated peaks: ', sorted(filtered_peaks_dir))

        return np.array(sorted(filtered_peaks_dir))

    def prepare_refinement(self, peaks_dir_indices: np.ndarray) \
            -> List[np.ndarray]:
        """ Prepare the directions along which direction and wavenumber finding will be done.
        """
        directions_ranges = []
        if peaks_dir_indices.size > 0:
            for peak_index in range(0, peaks_dir_indices.size):
                angles_half_range = self.local_estimator_params['ANGLE_AROUND_PEAK_DIR']
                direction_index = peaks_dir_indices[peak_index]
                tmp = np.arange(max(direction_index - angles_half_range, 0),
                                min(direction_index + angles_half_range + 1, 360)
                                )
                directions_range = self.radon_transforms[0].directions[tmp]
                directions_ranges.append(directions_range)

        # FIXME: what to do with opposite directions
        return directions_ranges

    def find_spectral_peaks(self,
                            directions_ranges: List[np.ndarray]) -> None:
        """ Find refined directions from the resampled cross correlation spectrum of the radon
        transforms of the 2 images and identify wavenumbers of the peaks along these directions.
        """
        for directions_range in directions_ranges:
            self._find_peaks_on_directions_range(self.full_linear_wavenumbers, directions_range)

        if self.debug_sample:
            self._metrics['kfft'] = self.full_linear_wavenumbers

    def _find_peaks_on_directions_range(self, wavenumbers: np.ndarray, directions: np.ndarray) \
            -> None:
        """ Find refined directions from the resampled cross correlation spectrum of the radon
        transforms of the 2 images and identify wavenumbers of the peaks along these directions.
        """
        metrics: Dict[str, Any] = {}
        # Detailed analysis of the signal for positive phase shifts
        self.radon_transforms[0].interpolate_sinograms_dfts(wavenumbers, directions)
        self.radon_transforms[1].interpolate_sinograms_dfts(wavenumbers, directions)
        sino1_fft = self.radon_transforms[0].get_sinograms_interpolated_dfts(directions)
        sino2_fft = self.radon_transforms[1].get_sinograms_interpolated_dfts(directions)
        phase_shift, spectrum_amplitude, sinograms_correlation_fft = \
            self._cross_correl_spectrum(sino1_fft, sino2_fft)
        total_spectrum = np.abs(phase_shift) * spectrum_amplitude
        max_heta = np.max(total_spectrum, axis=0)
        total_spectrum_normalized = max_heta / np.max(max_heta)

        peaks_freq = find_peaks(total_spectrum_normalized,
                                prominence=self.local_estimator_params['PROMINENCE_MULTIPLE_PEAKS'])
        peaks_freq = peaks_freq[0]
        peaks_wavenumbers_ind = np.argmax(total_spectrum[:, peaks_freq], axis=0)

        for index, direction_index in enumerate(peaks_freq):
            estimated_direction = directions[direction_index]
            wavenumber_index = peaks_wavenumbers_ind[index]
            estimated_phase_shift = phase_shift[wavenumber_index, direction_index]

            peak_sinogram = self.radon_transforms[0][estimated_direction]
            normalized_frequency = peak_sinogram.interpolated_dft_frequencies[wavenumber_index]
            wavenumber = normalized_frequency * self.radon_transforms[0].sampling_frequency

            energy = total_spectrum[wavenumber_index, direction_index]
            estimation = self.save_wave_field_estimation(estimated_direction, wavenumber,
                                                         estimated_phase_shift, energy)
            self.bathymetry_estimations.append(estimation)

        if self.debug_sample:

            metrics['max_heta'] = max_heta
            metrics['total_spectrum_normalized'] = total_spectrum_normalized
            metrics['sinograms_correlation_fft'] = sinograms_correlation_fft
            metrics['total_spectrum'] = total_spectrum
            self.metrics['kfft'] = wavenumbers
            self.metrics['totSpec'] = np.abs(total_spectrum) / np.mean(total_spectrum)
            self.metrics['interpolated_dft'] = metrics

    def save_wave_field_estimation(self, direction: float, wavenumber: float, phase_shift: float,
                                   energy: float) -> SpatialDFTBathyEstimation:
        """ Saves estimated parameters in a new estimation.

        :param direction: direction of the wave field (°)
        :param wavenumber: wavenumber of the wave field (m-1)
        :param phase_shift: phase difference estimated between the 2 images (rd)
        :param energy: energy of the wave field (definition TBD)
        :returns: a bathy estimation
        """
        wave_field_estimation = cast(SpatialDFTBathyEstimation,
                                     self.create_bathymetry_estimation(direction, 1 / wavenumber))

        wave_field_estimation.delta_phase = phase_shift
        wave_field_estimation.energy = energy
        return wave_field_estimation

    def _cross_correl_spectrum(self, sino1_fft: np.ndarray, sino2_fft: np.ndarray,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Computes the cross correlation spectrum of the radon transforms of the images, possibly
        restricted to a limited set of directions.

        :param sino1_fft: the DFT of the first sinogram, either standard or interpolated
        :param sino2_fft: the DFT of the second sinogram, either standard or interpolated
        :returns: A tuple of 2 numpy arrays and a dictionary with:
                  - the phase shifts
                  - the spectrum amplitude
                  - a dictionary containing intermediate results for debugging purposes
        """
        nb_wavenumbers = sino1_fft.shape[0]

        sinograms_correlation_fft = sino2_fft * np.conj(sino1_fft)
        phase_shift = np.angle(sinograms_correlation_fft)

        amplitude_sino1 = np.abs(sino1_fft) ** 2
        amplitude_sino2 = np.abs(sino2_fft) ** 2
        combined_amplitude = (amplitude_sino1 + amplitude_sino2)

        # Find maximum total energy per direction theta and normalize by the greater one
        spectrum_amplitude = np.abs(combined_amplitude) / (nb_wavenumbers**4)

        return phase_shift, spectrum_amplitude, sinograms_correlation_fft

    def get_full_linear_wavenumbers(self) -> np.ndarray:
        """  :returns: the requested sampling of the sinogram FFT
        """
        # frequencies based on wave characteristics:
        period_samples = np.arange(self.global_estimator.waves_period_max,
                                   self.global_estimator.waves_period_min,
                                   -self.local_estimator_params['STEP_T'])
        return cast(np.ndarray, wavenumber_offshore(period_samples, self.gravity))
