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
        self._celerity = np.nan
        self.register_wavelength_change(self.wavelength_has_changed)

    def wavelength_has_changed(self) -> None:
        """ When wavelength has changed (new value is ensured to be different from the previous one)
        either reset period and celerity if both were set, or update one of them if the other is set
        """
        if not np.isnan(self.period) and not np.isnan(self.celerity):
            self._period = np.nan
            self._celerity = np.nan
        self._solve_movement_equation()

    def _solve_movement_equation(self) -> None:
        """ Solves the movement equation ( L=c*T ) when one of the 3 variables is not set and the 2
        other are set. In other cases (all 3 variables set or more than 1 variable not set)
        do not change anything.
        """
        wavelength_set = not np.isnan(self.wavelength)
        period_set = not np.isnan(self.period)
        celerity_set = not np.isnan(self.celerity)
        if wavelength_set and period_set and not celerity_set:
            self._celerity = self.wavelength / self.period
        elif wavelength_set and not period_set and celerity_set:
            self._period = self.wavelength / self.celerity
        elif not wavelength_set and period_set and celerity_set:
            self.wavelength = self.celerity * self.period

    @property
    def period(self) -> float:
        """ :returns: The waves field period (s), which was either externally provided or computed
        from the wavelength and the celerity
        """
        return self._period

    @period.setter
    def period(self, value: float) -> None:
        if value != self._period:
            if value < 0.:
                self.invert_direction()
            self._period = abs(value)
            if not np.isnan(self.celerity) and not np.isnan(self.wavelength):
                self._celerity = np.nan
                self.wavelength = np.nan
            self._solve_movement_equation()

    @property
    def celerity(self) -> float:
        """ :returns: The waves field velocity (m/s), which was either externally provided or
        computed from the wavelength and the period
        """
        return self._celerity

    @celerity.setter
    def celerity(self, value: float) -> None:
        if value != self.celerity:
            if value < 0.:
                self.invert_direction()
            self._celerity = abs(value)
            if not np.isnan(self.period) and not np.isnan(self.wavelength):
                self._period = np.nan
                self.wavelength = np.nan
            self._solve_movement_equation()

    def __str__(self) -> str:
        result = WavesFieldSampleGeometry.__str__(self)
        result += f'\nDynamics:   period: {self.period:5.2f} (s)  celerity: {self.celerity:5.2f}'
        return result
