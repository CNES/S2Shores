# -*- coding: utf-8 -*-
""" Class encapsulating operations on the radon transform of an image for waves processing

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from functools import lru_cache

from typing import Optional, List  # @NoMove

import numpy as np  # @NoMove


from ..generic_utils.image_filters import circular_masking
from ..generic_utils.symmetric_radon import symmetric_radon
from .sinograms import Sinograms
from .waves_image import WavesImage
from .waves_sinogram import WavesSinogram


DEFAULT_ANGLE_MIN = -180.
DEFAULT_ANGLE_MAX = 180.


def linear_directions(angle_min: float, angle_max: float, directions_step: float) -> np.ndarray:
    return np.linspace(angle_min, angle_max,
                       int((angle_max - angle_min) / directions_step),
                       endpoint=False)


@lru_cache()
def sinogram_weights(nb_samples: int) -> np.ndarray:
    """ Computes weighting function to account for less energy at the extremities of a sinogram.

    :param nb_samples: the number of samples in the sinogram (its length)
    :return: weighting function with extremities modified to be non-zero

    """
    samples = np.linspace(-1., 1., nb_samples)
    weights = 1. / np.sqrt(1. - samples**2)
    weights[0] = weights[1]
    weights[-1] = weights[-2]
    return weights


class WavesRadon(Sinograms):
    """ Class handling the Radon transform of some image.
    """

    def __init__(self, image: WavesImage, selected_directions: Optional[np.ndarray] = None,
                 directions_step: float = 1.) -> None:
        """ Constructor

        :param image: a 2D array containing an image
        :param selected_directions: a set of directions onto which the radon transform must be
                                    provided. If unspecified, all integre angles between -180° and
                                    +180° are considered.
        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
        """
        self.pixels = circular_masking(image.pixels)

        # TODO: Quantize directions when selected_directions is provided?
        if selected_directions is None:
            selected_directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
                                                    directions_step)

        radon_transform_list = self._compute(selected_directions)

        super().__init__()
        self.quantization_step = directions_step
        self.sampling_frequency = image.sampling_frequency
        self.insert_sinograms(radon_transform_list, selected_directions)

    def _compute(self, selected_directions: np.ndarray) -> List[WavesSinogram]:
        """ Compute the radon transform of the image over a set of directions
        """
        # FIXME: quantization may imply that radon transform is not computed on stored directions
        radon_transform = symmetric_radon(self.pixels, theta=selected_directions)

        sinograms: List[WavesSinogram] = []
        for index, _ in enumerate(selected_directions):
            sinogram = WavesSinogram(radon_transform[:, index])
            sinograms.append(sinogram)
        return sinograms
