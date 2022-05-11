# -*- coding: utf-8 -*-
""" Class encapsulating an image onto which wave field estimation will be made


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 7 mars 2021
"""
from typing import Tuple, Callable, List, Any

import numpy as np

from ..generic_utils.numpy_utils import circular_mask
from ..image.image_geometry_types import ImageWindowType


ImageProcessingFilters = List[Tuple[Callable, List[Any]]]


class WavesImage:
    def __init__(self, pixels: np.ndarray, resolution: float) -> None:
        """ Constructor

        :param pixels: a 2D array containing an image over water
        :param resolution: Image resolution in meters
        """
        self.resolution = resolution

        # FIXME: introduced until there is a true image versions management
        self.original_pixels = pixels.copy()
        self.pixels = pixels

    def apply_filters(self, processing_filters: ImageProcessingFilters) -> 'WavesImage':
        """ Apply filters on the image pixels and return a new WavesImage

        :param processing_filters: A list of functions together with their parameters to apply
                                   sequentially to the image pixels.
        :returns: a WavesImage with the result of the filters application
        """
        result = self.original_pixels.copy()
        for processing_filter, filter_parameters in processing_filters:
            result = processing_filter(result, *filter_parameters)
        return WavesImage(result, self.resolution)

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
        # FIXME: Ratio of the disk area on the chip area should be closer to PI/4 (0.02 difference)
        return circular_mask(self.pixels.shape[0], self.pixels.shape[1], self.pixels.dtype)

    def extract_sub_image(self, window: ImageWindowType) -> 'WavesImage':
        """ :param window: the window to extract from this image
        :returns: a new WavesImage defined by a window inside this image.
        """
        return WavesImage(self.pixels[window[0]:window[1] + 1, window[2]:window[3] + 1],
                          self.resolution)

    def __str__(self) -> str:
        return f'Resolution: {self.resolution}  Shape: {self.pixels.shape}:\n{self.pixels}'
