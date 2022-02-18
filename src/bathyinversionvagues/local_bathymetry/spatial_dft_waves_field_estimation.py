# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample.

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 sep 2021
"""
import numpy as np

from ..data_model.waves_field_estimation import WavesFieldEstimation


class SpatialDFTWavesFieldEstimation(WavesFieldEstimation):
    """ This class encapsulates the information estimated in a waves field sample by a
    SpatialDFTBathyEstimator.

    It defines the estimation attributes specific to this estimator.
    """

    def __init__(self, gravity: float, depth_estimation_method: str) -> None:

        super().__init__(gravity, depth_estimation_method)

        self._delta_phase_ratio = np.nan
        self._energy = np.nan

    @property
    def delta_celerity(self) -> float:
        # FIXME: define this quantity
        """ :returns: TBD """
        return np.nan

    @property
    def delta_phase_ratio(self) -> float:
        # FIXME: define this quantity
        """ :returns: the ratio of the phase difference compared to TBD """
        return self._delta_phase_ratio

    @delta_phase_ratio.setter
    def delta_phase_ratio(self, value: float) -> None:
        self._delta_phase_ratio = value

    @property
    def energy(self) -> float:
        """ :returns: the energy of the waves field """
        return self._energy

    @energy.setter
    def energy(self, value: float) -> None:
        self._energy = value

    @property
    def energy_ratio(self) -> float:
        """ :returns: The ratio of energy relative to the max peak """
        return (self.delta_phase_ratio ** 2) * self.energy

    def __str__(self) -> str:
        result = WavesFieldEstimation.__str__(self)
        result += f'\n    delta phase: {self.delta_phase:5.2f} (rd)'
        result += f'  delta phase ratio: {self.delta_phase_ratio:5.2f} '
        result += f'\n    energy: {self.energy:5.2f} (???)'
        result += f'  energy ratio: {self.energy_ratio:5.2f} '
        return result
