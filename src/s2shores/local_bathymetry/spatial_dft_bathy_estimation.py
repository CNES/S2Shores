# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample.

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 sep 2021
"""
from typing import Tuple

import numpy as np

from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation
from ..image_processing.waves_radon import DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX, \
                                           DEFAULT_ANGLE_STEP


class SpatialDFTBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a wave field sample by a
    SpatialDFTBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """

    def __init__(self, gravity: float, depth_estimation_method: str,
                 period_range: Tuple[float, float], linearity_range: Tuple[float, float],
                 shallow_water_limit: float) -> None:

        super().__init__(gravity, depth_estimation_method, period_range, linearity_range,
                         shallow_water_limit)

        self._energy = np.nan
        #TODO : waves_spectrum should be set to an array of nan, but dimension 
        #       (nb_directions, nb_frequencies) are unaccessible.
        self._waves_spectrum = None

    @property
    def delta_celerity(self) -> float:
        # FIXME: define this quantity
        """ :returns: TBD """
        return np.nan

    @property
    def energy(self) -> float:
        """ :returns: the energy of the wave field """
        return self._energy

    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = value

    @property
    def energy_ratio(self) -> float:
        """ :returns: The ratio of energy relative to the max peak """
        return (self.relative_period ** 2) * self.energy

    @property
    def waves_spectrum(self) -> np.ndarray:
        """ :returns: The wave spectrum """
        return self._waves_spectrum
    
    @waves_spectrum.setter
    def waves_spectrum(self, value: np.ndarray) -> None:
        self._waves_spectrum = value

    def __str__(self) -> str:
        result = super().__str__()
        result += f'\n    energy: {self.energy:5.2f} (???)'
        result += f'  energy ratio: {self.energy_ratio:5.2f} '
        return result
