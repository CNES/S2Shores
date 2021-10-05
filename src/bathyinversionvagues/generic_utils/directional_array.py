# -*- coding: utf-8 -*-
""" Definition of the DirectionalArray class and associated functions

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 4 mars 2021
"""
from typing import Optional, Union, Tuple  # @NoMove

import numpy as np
import numpy.typing as npt

DEFAULT_ANGLE_MIN = -180.
DEFAULT_ANGLE_MAX = 0.
DEFAULT_DIRECTIONS_STEP = 1.


def normalize_direction(direction: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # direction between 0 and 360 degrees
    normalized_direction = direction % 360.
    # direction between -180 and +180 degrees
    if isinstance(normalized_direction, float):
        if normalized_direction >= 180.:
            normalized_direction -= 360.
    else:
        normalized_direction[normalized_direction >= 180.] -= 360.

    return normalized_direction


def linear_directions(angle_min: float, angle_max: float, directions_step: float) -> np.ndarray:
    return np.linspace(angle_min, angle_max,
                       int((angle_max - angle_min) / directions_step),
                       endpoint=False)


# TODO: add a "symmetric" property, allowing to consider directions modulo pi as equivalent
# TODO: enable ordering such that circularity can be exposed (around +- 180° and 0°)
class DirectionalArray:
    def __init__(self, array: np.ndarray, directions: np.ndarray,
                 directions_step: float = DEFAULT_DIRECTIONS_STEP) -> None:
        """ Constructor

        :param array: a 2D array containing directional vectors along each column
        :param directions: the set of directions in degrees associated to each array column.
        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
        :raises TypeError: when array or directions have not the right number of dimensions
        :raises ValueError: when the number of dimensions is not consistent with the number of
                            columns in the array.
        """
        # Check that numpy arguments are of the right dimensions when provided
        if array.ndim != 2:
            raise TypeError('array for a DirectionalArray must be a 2D numpy array')

        if directions.ndim != 1:
            raise TypeError('dimensions for a DirectionalArray must be a 1D numpy array')

        if directions.size != array.shape[1]:
            raise ValueError('directions size must be equal to the number of columns of the array')

        self._directions_step = directions_step

        quantized_directions = self._prepare_directions(directions)
        if quantized_directions.shape[0] != array.shape[1]:
            raise ValueError('dimensions argument has not the same number of elements '
                             f'({quantized_directions.shape[0]}) than the number '
                             f'of columns in the array ({array.shape[1]})')

        self._array = array
        # TODO: implement the directions as the keys of a dictionary pointing to views in the array?
        self._directions = quantized_directions

    @classmethod
    def create_empty(cls,
                     height: int,
                     directions: Optional[np.ndarray] = None,
                     directions_step: float = DEFAULT_DIRECTIONS_STEP,
                     dtype: npt.DTypeLike = np.float64) -> 'DirectionalArray':
        """ Creation of an empty DirectionalArray

        :param directions_step: the step to use for quantizing direction angles, for indexing
                                purposes. Direction quantization is such that the 0 degree direction
                                is used as the origin, and any direction angle is transformed to the
                                nearest quantized angle for indexing that direction in the radon
                                transform.
        :raises TypeError: when directions is not a 1D array
        """
        # Check that directions argument
        if directions is None:
            directions = linear_directions(DEFAULT_ANGLE_MIN, DEFAULT_ANGLE_MAX, directions_step)
        elif directions.ndim != 1:
            raise TypeError('dimensions for a DirectionalArray must be a 1D numpy array')

        array = np.empty((height, directions.size), dtype=dtype)

        return cls(array, directions=directions, directions_step=directions_step)

    # TODO: remove this property and use get_as_array instead
    @property
    def array(self) -> np.ndarray:
        """ :return: the array of this DirectionalArray """
        return self._array

    @property
    def directions(self) -> np.ndarray:
        """ :return: the directions defined in this DirectionalArray """
        return self._directions

    @property
    def nb_directions(self) -> int:
        """ :return: the number of directions defined in this DirectionalArray"""
        return self.array.shape[1]

    @property
    def height(self) -> int:
        """ :return: the height of each directional vector in this DirectionalArray"""
        return self.array.shape[0]

    def _prepare_directions(self, directions: np.ndarray) -> np.ndarray:
        """ Quantize a set of directions and verify that no duplicates are created by quantization

        :param directions: an array of directions
        :returns: an array of quantized directions
        :raises ValueError: when the number of elements in the quantized array is different from
                            the number of elements in the input directions array
        """
        # TODO: consider defining direction_step from dimensions contents before quantizing
        # TODO: consider reordering the directions in increasing order
        quantized_directions = self._quantize_direction(directions)[0]
        unique_directions = np.unique(quantized_directions)
        if unique_directions.size != directions.size:
            raise ValueError('some dimensions values are too close to each other considering '
                             f'the dimensions quantization step: {self._directions_step}°')
        return quantized_directions

    def _quantize_direction(self, direction: Union[float, np.ndarray]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        # direction between -180 and +180 degrees
        normalized_direction = normalize_direction(direction)

        index_direction = np.around(normalized_direction / self._directions_step)
        quantized_direction = index_direction * self._directions_step

        return quantized_direction, index_direction

    # TODO: allow float or array, return 1D or 2D array
    def values_for(self, direction: float) -> np.ndarray:
        direction_index = self._find_index(direction)
        return self._array[:, direction_index]

    def values_at_index(self, direction_index: int) -> np.ndarray:
        return self.array[:, direction_index]

    def set_at_direction(self, direction: float, array: np.ndarray) -> None:
        direction_index = self._find_index(direction)
        self.array[:, direction_index] = array

    def _find_index(self, direction: float) -> int:
        quantized_direction = self._quantize_direction(direction)
        direction_indexes = np.where(self.directions == quantized_direction[0])
        if direction_indexes[0].size != 1:
            raise ValueError(f'direction {direction} not found in the directional array')
        return direction_indexes[0]

    def get_as_array(self, directions: Optional[np.ndarray] = None) -> np.ndarray:
        """ Returns a 2D array with the requested directional values as columns

        :param directions: a vector of directions to store in the returned array, in their order
        :returns: a 2D array with the requested directional values as columns
        """
        if directions is None:
            return self._array
        quantized_directions, _ = self._quantize_direction(directions)
        # Build array by selecting the requested directions
        array_excerpt = np.empty((self.height, quantized_directions.size))
        for i, direction in enumerate(quantized_directions):
            array_excerpt[:, i] = self.values_for(direction).reshape(self.height)
        return array_excerpt
