# -*- coding: utf-8 -*-
"""
module -- Class encapsulating an image onto which waves estimation will be made


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 7 mars 2021
"""
import numpy as np

from .numpy_utils import circular_mask
from .shoresutils import funDetrend_2d


# TODO: add the management of the image position
# TODO: add the azimuth of the image, if known
# TODO: add the possibility to apply several preprocessing filters
class WavesImage():
    def __init__(self, pixels: np.ndarray, satellite: str,
                 resolution: float = 10., detrend: bool=True) -> None:
        """ Constructor

        :param pixels: a 2D array containing an image over water
        :param resolution: Image resolution in meters
        :param detrend: True (default) if an optional detrend must be applied on the image.
        """
        self.pixels = pixels
        self.pixels = self.pixels * self.circle_image
        if detrend:
            self.pixels = funDetrend_2d(pixels)
        self.resolution = resolution

    @property
    def sampling_frequency(self) -> float:
        """ :returns: The spatial sampling frequency of this image (m-1)"""
        return 1. / self.resolution

    @property
    def energy(self) -> float:
        """ :returns: The energy of the image"""
        return np.sum(self.pixels * self.pixels)

    @property
    def energy_inner_disk(self) -> np.ndarray:
        """ :returns: The energy of the image within its inscribed disk"""

        return np.sum(self.pixels * self.pixels * self.circle_image)

    @property
    def circle_image(self) -> np.ndarray:
        """ :returns: The inscribed disk"""
        return circular_mask(self.pixels.shape[0], self.pixels.shape[1], self.pixels.dtype)
