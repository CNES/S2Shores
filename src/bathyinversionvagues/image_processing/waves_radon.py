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
from ..generic_utils.numpy_utils import HashableNdArray
from ..generic_utils.symmetric_radon import symmetric_radon
from .sinograms import Sinograms
from .waves_image import WavesImage
from .waves_sinogram import WavesSinogram


DEFAULT_ANGLE_MIN = -180.
DEFAULT_ANGLE_MAX = 180.
DEFAULT_ANGLE_STEP = 1.


def linear_directions(angle_min: float, angle_max: float, angles_step: float) -> np.ndarray:
    """ Computes a sampling of a range of directions suitable for a Radon transform.

    :param angle_min: the lowest direction angle of the range
    :param angle_max: the highest direction angle of the range
    :param angles_step: the step between the directions
    :returns: An array with angles evenly spaced between angle_min and angle_max (excluded)
    """
    return np.linspace(angle_min, angle_max,
                       int((angle_max - angle_min) / angles_step),
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


@lru_cache()
def directional_circle_factors(nb_samples: int, directions: HashableNdArray) -> np.ndarray:
    """ Computes a set of correction factors which can be applied to each direction of a Radon
    transform to take into account the quantization of the circle boundary which leads to
    different lengths of integration along each direction.

    :param nb_samples: the size of the Radon transform along one dimension
    :param directions: a set of angles for which the correction factors must be computed
    :returns: an array providing the relative length of the intersection of a direction with the
              quantized circle onto which the Radon transform is computed.
    """
    ones_square = np.ones((nb_samples, nb_samples))
    ones_disk = circular_masking(ones_square)
    ones_disk_radon_transform = symmetric_radon(ones_disk, theta=directions.unwrap())
    ones_disk_directions_sums = np.sum(ones_disk_radon_transform, axis=0)
    return ones_disk_directions_sums / ones_disk_directions_sums[0]


class WavesRadon(Sinograms):
    """ Class handling the Radon transform of some image.
    """

    def __init__(self, image: WavesImage, selected_directions: Optional[np.ndarray] = None,
                 directions_quantization: Optional[float] = None) -> None:
        """ Constructor

        :param image: a 2D array containing an image
        :param selected_directions: a set of directions onto which the radon transform must be
                                    provided. If unspecified, all integre angles between -180° and
                                    +180° are considered.
        :param directions_quantization: the step to use for quantizing direction angles, for
                                        indexing purposes. Direction quantization is such that the
                                        0 degree direction is used as the origin, and any direction
                                        angle is transformed to the nearest quantized angle for
                                        indexing that direction in the radon transform.
        """
        super().__init__(1. / image.resolution, directions_quantization)

        self.pixels = circular_masking(image.pixels.copy())

        # TODO: Quantize directions when selected_directions is provided?
        if selected_directions is None:
            selected_directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX,
                                                    DEFAULT_ANGLE_STEP)

        radon_transform = symmetric_radon(self.pixels, theta=selected_directions)

        sinograms: List[WavesSinogram] = []
        for index, _ in enumerate(selected_directions):
            sinogram = WavesSinogram(radon_transform[:, index])
            sinograms.append(sinogram)

        self.insert_sinograms(sinograms, selected_directions)
