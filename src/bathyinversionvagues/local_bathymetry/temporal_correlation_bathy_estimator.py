# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 10:12:00 2021

Module containing all wave parameters estimation methods

@author: erwinbergsma
         gregoirethoumyre
         degoulromain
"""
from typing import List, Optional

import numpy as np

from ..image_processing.shoresutils import (fft_filtering, compute_sinogram,
                                            create_sequence_time_series_temporal,
                                            compute_temporal_correlation, compute_celerity,
                                            cartesian_projection, correlation_tuning,
                                            sinogram_tuning, compute_wave_length, compute_period,
                                            temporal_reconstruction,
                                            temporal_reconstruction_tuning,
                                            )
from ..image_processing.waves_image import WavesImage
from ..waves_fields_display import draw_results

from .wavemethods import build_correlation_output


def temporal_correlation_method(images_sequence: List[WavesImage], global_estimator):
    """
    Bathymetry computation function based on time series correlation

    Parameters
    ----------
    Im : numpy.ndarray
        Sub-windowed images in M x N x BANDS
     Returns
    -------
    dict:
        As output we deliver a dictionary containing
            -   cel     =   Wave celerity               [m/s]
            -   nu       =   linear Wave number                 [1/m]
            -   L       =   Wavelength                  [m]
            -   T       =   Approximate wave period     [sec]
            -   dir     =   Wave direction (RADON)      [degrees]

    """
    config = global_estimator.waveparams
    params = config.TEMPORAL_METHOD
    # FIXME: temporary adaptor before getting rid of stacked np.ndarrays.
    Im = np.dstack([image.pixels for image in images_sequence])

    try:
        if params.PASS_BAND_FILTER:
            Im, flag = fft_filtering(Im, params.RESOLUTION.SPATIAL,
                                     config.PREPROCESSING.PASSBAND.HIGH_PERIOD,
                                     config.PREPROCESSING.PASSBAND.LOW_PERIOD, 9.81)
        stime_series, xx, yy = create_sequence_time_series_temporal(Im=Im,
                                                                    percentage_points=params.PERCENTAGE_POINTS)
        corr = compute_temporal_correlation(sequence_thumbnail=stime_series,
                                            number_frame_shift=params.TEMPORAL_LAG)
        corr_car, distances, angles = cartesian_projection(corr_matrix=corr, xx=xx, yy=yy,
                                                           spatial_resolution=params.RESOLUTION.SPATIAL)
        corr_car_tuned = correlation_tuning(correlation_matrix=corr_car,
                                            ratio=params.TUNING.RATIO_SIZE_CORRELATION)
        (sinogram_max_var, angle, variance, radon_matrix) = compute_sinogram(correlation_matrix=corr_car_tuned,
                                                                             median_filter_kernel_ratio=params.TUNING.MEDIAN_FILTER_KERNEL_RATIO_SINOGRAM,
                                                                             mean_filter_kernel_size=params.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        sinogram_tuned = sinogram_tuning(sinogram=sinogram_max_var,
                                         mean_filter_kernel_size=params.TUNING.MEAN_FILTER_KERNEL_SIZE_SINOGRAM)
        wave_length, wave_length_peaks = compute_wave_length(sinogram=sinogram_tuned)
        celerity, argmax = compute_celerity(sinogram=sinogram_tuned, wave_length=wave_length,
                                            spatial_resolution=params.RESOLUTION.SPATIAL,
                                            time_resolution=params.RESOLUTION.TEMPORAL,
                                            temporal_lag=params.TEMPORAL_LAG)
        SS = temporal_reconstruction(angle=angle, angles=np.degrees(angles), distances=distances, celerity=celerity,
                                     correlation_matrix=corr,
                                     time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION)
        SS_filtered = temporal_reconstruction_tuning(SS,
                                                     time_interpolation_resolution=params.RESOLUTION.TIME_INTERPOLATION,
                                                     low_frequency_ratio=params.TUNING.LOW_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION,
                                                     high_frequency_ratio=params.TUNING.HIGH_FREQUENCY_RATIO_TEMPORAL_RECONSTRUCTION)
        T, peaks_max = compute_period(SS_filtered=SS_filtered,
                                      min_peaks_distance=params.TUNING.MIN_PEAKS_DISTANCE_PERIOD)
        if params.DEBUG_MODE:
            draw_results(Im, angle, corr_car, radon_matrix, variance, sinogram_max_var, sinogram_tuned, argmax,
                         wave_length_peaks, wave_length, params, celerity, peaks_max, SS_filtered, T)

        waves_field_estimation, waves_fieldestimation_as_dict = build_correlation_output(
            angle, wave_length, T, celerity, config)

        return waves_fieldestimation_as_dict

    except Exception as excp:
        print(f'Bathymetry computation failed: {str(excp)}')
        return {}
