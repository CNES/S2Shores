# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
from typing import cast, Tuple
import numpy as np

from ..bathy_physics import time_sampling_factor_offshore, time_sampling_factor_low_depth
from .waves_field_sample_bathymetry import WavesFieldSampleBathymetry
from .waves_field_sample_estimation import WavesFieldSampleEstimation


class WavesFieldEstimation(WavesFieldSampleEstimation, WavesFieldSampleBathymetry):
    """ This class encapsulates the information estimating a waves field sample.

    It inherits from WavesFieldSampleBathymetry and defines specific attributes related to the
    sample estimation based on physical bathymetry.
    """

    def __init__(self, gravity: float, depth_estimation_method: str,
                 period_range: Tuple[float, float], linearity_range: Tuple[float, float],
                 shallow_water_limit: float) -> None:

        WavesFieldSampleEstimation.__init__(self, gravity, period_range, shallow_water_limit)
        WavesFieldSampleBathymetry.__init__(self, gravity, period_range, depth_estimation_method,
                                            linearity_range)

    def is_physical(self) -> bool:
        """  Check if a waves field estimation satisfies physical constraints.

        :returns: True is the waves field is valid, False otherwise
        """
        return (self.is_period_valid() and
                self.is_time_sampling_factor_valid() and
                self.is_linearity_valid())

    @property
    def delta_phase_ratio(self) -> float:
        """ :returns: the fraction of the maximum phase shift allowable in deep waters """
        time_sampling_offshore = cast(float,
                                      time_sampling_factor_offshore(self.wavenumber,
                                                                    self.delta_time,
                                                                    self.gravity))
        return self.delta_phase / (2 * np.pi * time_sampling_offshore)

    def __str__(self) -> str:
        result = WavesFieldSampleEstimation.__str__(self)
        result += '\n' + WavesFieldSampleBathymetry.__str__(self)
        result += f'\nBathymetry Estimation:  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        return result
