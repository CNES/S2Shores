# -*- coding: utf-8 -*-
""" Module gathering all image filters which can be applied on a 1D numpy array

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021-2024 CNES. All rights reserved.
:license: see LICENSE file
:created: 26 August 2021
"""
from scipy.signal import medfilt, detrend, butter, lfilter

import numpy as np


def filter_mean(array: np.ndarray, window: int) -> np.ndarray:
    """ Performs mean filter on a signal

    :param array: entry signal
    :param window: size of the moving average window
    :returns: filtered array
    :raises ValueError: when the array is too small compared to window size
    """
    if len(array) < 2 * window:
        raise ValueError('array is too small compared to the window')
    padded_array = np.concatenate((np.full(window, np.mean(array[:window])),
                                   array,
                                   np.full(window, np.mean(array[-(window + 1):]))))
    return np.convolve(padded_array, np.ones(2 * window + 1) / (2 * window + 1), 'valid')


def remove_median(array: np.ndarray, kernel_ratio: float) -> np.ndarray:
    """ Performs median removal on a signal

    :param array: entry signal
    :param kernel_ratio: ratio size of the median kernel compared to the signal
    :returns: filtered array
    """
    kernel_size = round(len(array) * kernel_ratio)
    if (kernel_size % 2) == 0:
        kernel_size = kernel_size + 1
    return array - filter_median(array, kernel_size)

def filter_median(array: np.ndarray, kernel_size: int) -> np.ndarray:
    """ Perform median filtering on a signal

    :param array: entry signal
    :param kernel_size: size of the median kernel filtering the signal
    :returns: filtered array
    """
    return medfilt(array, kernel_size)

def detrend_signal(array: np.ndarray, axis:int=0) -> np.ndarray:
    """Performs a detrend process on a signal

    :param array: entry signal
    :param axis: axis on which the detrend is applied (default axis=0)
    :returns: detrended array
    """
    return detrend(array, axis=axis)

def butter_bandpass_filter(array: np.ndarray, lowcut_period: float, highcut_period: float,
                           fs: float, axis: int=0) -> np.ndarray:
    """ Performs a Band-pass filtering using a Butterworth filter

    :param array: entry signal
    :param lowcut_period: signal period in seconds defining low frequency cut-off
    :param highcut_period: signal period in seconds defining high frequency cut-off
    :param fs: signal sampling frequency (Hz)
    :param axis: axis on which the filtering is applied (default axis=0)
    :returns: filtered array
    """
    b, a = butter_bandpass(lowcut_period, highcut_period, fs)
    filtered_array = lfilter(b, a, array, axis=axis)
    return filtered_array

def butter_bandpass(lowcut_period: float, highcut_period: float, fs: float, order: int=5)-> list:
    """ Compute the BP Butterworth filter coefficients

    :param lowcut_period: signal period in seconds defining low frequency cut-off
    :param highcut_period: signal period in seconds defining high frequency cut-off
    :param fs: signal sampling frequency (Hz)
    :param order: fitler order (default=5)
    :returns: filter coefficients
    """
    nyq = 0.5 * fs
    low = (1/lowcut_period) / nyq
    high = (1/highcut_period) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
