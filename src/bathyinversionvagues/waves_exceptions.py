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
