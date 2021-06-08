# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 6 mars 2021
"""
import numpy as np

from .waves_field_sample_geometry import WavesFieldSampleGeometry


class WavesFieldSampleDynamics(WavesFieldSampleGeometry):
    """ This class encapsulates the information related to the dynamics of a waves field sample.
    It inherits from WavesFieldSampleGeometry which describes the observed field geometry,
    and contains specific attributes related to the field dynamics:

    - its period
    - its celerity

    """

    def __init__(self) -> None:
        super().__init__()
        self._period = np.nan

    @property
    def period(self) -> float:
        """ :returns: The waves field period (s) """
        return self._period

    @period.setter
    def period(self, value: float) -> None:
        self._period = value

    @property
    def celerity(self) -> float:
        """ :returns: The waves field velocity (m/s) """
        return 1. / (self.wavenumber * self.period)

    def __str__(self) -> str:
        result = WavesFieldSampleGeometry.__str__(self)
        result += f'\nperiod: {self.period:5.2f} (s)  celerity: {self.celerity:5.2f}'
        return result
