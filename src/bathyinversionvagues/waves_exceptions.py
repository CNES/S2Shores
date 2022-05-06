# -*- coding: utf-8 -*-
""" Exceptions used in bathymetry estimation

:author: GIROS Alain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 20 mai 2021
"""
from typing import Optional


class WavesException(Exception):
    """ Base class for all waves estimation exceptions
    """

    def __init__(self, reason: Optional[str] = None) -> None:
        super().__init__()
        self.reason = reason

    def __str__(self) -> str:
        if self.reason is None:
            return ''
        return f'{self.reason}'


class WavesEstimationError(WavesException):
    """ Exception raised when an error occurs in bathymetry estimation
    """


class SequenceImagesError(WavesException):
    """ Exception raised when sequence images can not be properly exploited
    """


class NotExploitableSinogram(WavesException):
    """ Exception raised when sinogram can not be exploited
    """


class CorrelationComputationError(WavesException):
    """ Exception raised when correlation can not be computed
    """


class DebugDisplayError(WavesException):
    """ Exception raised when debug display fails
    """


class ProductNotFound(WavesException):
    """ Exception raised when a product cannot be found
    """


class WavesIndexingError(WavesException):
    """ Exception raised when a point cannot be found in the sampling with its coordinates.
    """


class WavesEstimationAttributeError(WavesException):
    """ Exception raised when an attribute is not available in an estimation.
    """
