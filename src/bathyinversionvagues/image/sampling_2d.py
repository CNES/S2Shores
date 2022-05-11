# -*- coding: utf-8 -*-
""" Definition of the Sampling2D class and associated functions

:author: GIROS Alain
:created: 05/05/2021
"""
from typing import Tuple, List, Iterator  # @NoMove

import numpy as np  # @NoMove
from shapely.geometry import Polygon, Point

from ..generic_utils.numpy_utils import split_samples
from ..waves_exceptions import WavesIndexingError


class Sampling2D:
    """ A 2S sampling is a subset of samples in a 2D space. It is built by taking consecutive
    samples in some samples coordinates lists, which means that there is no constraint on the
    spatial distribution of these samples. It is up to the caller to impose these constraints
    by providing increasing or decreasing ordered lists of coordinates or whatever desired order,
    according to the needs.
    """

    def __init__(self, x_samples: np.ndarray, y_samples: np.ndarray) -> None:
        """ Define the samples defining this 2D sampling. These samples correspond to the cross
        product of the X and Y coordinates.

        :param x_samples: the X coordinates defining the 2D sampling
        :param y_samples: the Y coordinates defining the 2D sampling
        """

        self._x_samples = x_samples
        self._y_samples = y_samples

    @property
    def x_samples(self) -> np.ndarray:
        """ :returns: the sampling coordinates along the X axis.
        """
        return self._x_samples

    @property
    def y_samples(self) -> np.ndarray:
        """ :returns: the sampling coordinates along the Y axis.
        """
        return self._y_samples

    @property
    def upper_left_sample(self) -> Point:
        """ :returns: The upper left sample of this sampling, assuming that
                      Y axis is decreasing from top to down.
        """
        return Point(self._x_samples[0], self._y_samples[-1])

    @property
    def lower_right_sample(self) -> Point:
        """ :returns: the loxer right sample of this sampling, assuming that
                      Y axis is decreasing from top to down.
        """
        return Point(self._x_samples[-1], self._y_samples[0])

    @property
    def nb_samples(self) -> int:
        """ :returns: the number of samples in this 2D sampling.
        """
        return len(self._x_samples) * len(self._y_samples)

    @property
    def shape(self) -> Tuple[int, int]:
        """ :returns: the shape of the 2D sampling.
        """
        return self._y_samples.shape[0], self._x_samples.shape[0]

    def index_point(self, point: Point) -> Tuple[int, int]:
        """ Retrieve the indexes of the coordinates of a point in the X and Y samples

        :param point: a point in 2D, whose coordinates must be retrieved in the sampling
        :returns: the indexes of X and Y in the sampling definitions
        :raises WavesIndexingError: when one coordinate of the point is undefined in the sampling
        """
        x_index = np.where(self._x_samples == point.x)
        if x_index[0].size == 0:
            msg_err = f'X coordinate: { point.x} undefined in x_samples: {self._x_samples}'
            raise WavesIndexingError(msg_err)
        y_index = np.where(self._y_samples == point.y)
        if y_index[0].size == 0:
            msg_err = f'Y coordinate: { point.y} undefined in y_samples: {self._y_samples}'
            raise WavesIndexingError(msg_err)
        return x_index[0][0], y_index[0][0]

    def x_y_sampling(self) -> Iterator[Point]:
        """ A generator returning all points in this Sampling2D one after the other. Sampling is
        done by providing all the points for the first X coordinate and then all the points for the
        next X coordinate, and so on.

        :yields: successive points in the sampling
        """
        for x_sample in self._x_samples:
            for y_sample in self._y_samples:
                yield Point(x_sample, y_sample)

    def y_x_sampling(self) -> Iterator[Point]:
        """ A generator returning all points in this Sampling2D one after the other. Sampling is
        done by providing all the points for the first Y coordinate and then all the points for the
        next Y coordinate, and so on.

        :yields: successive points in the sampling
        """
        for y_sample in self._y_samples:
            for x_sample in self._x_samples:
                yield Point(x_sample, y_sample)

    def split(self, nb_tiles_max: int = 1) -> List['Sampling2D']:
        """ Build a set of sampling2d which makes a tiling of this 2D sampling. The number of
        created tiles may be lower than the requested number of tiles, if this number is not a
        square number.

        :param nb_tiles_max: the maximum number of tiles in the tiling

        :returns: a list of samplings defining a set of tiles covering this sampling.
        """
        tiles_samplings = []
        # Full samples cropped in crop*crop tiles
        crop = int(np.sqrt(nb_tiles_max))
        nb_tiles_x = min(crop, self.x_samples.size)
        nb_tiles_y = min(crop, self.y_samples.size)

        x_samples_parts = split_samples(self.x_samples, nb_tiles_x)
        y_samples_parts = split_samples(self.y_samples, nb_tiles_y)
        for x_samples_part in x_samples_parts:
            for y_samples_part in y_samples_parts:
                tiles_samplings.append(Sampling2D(x_samples_part, y_samples_part))
        return tiles_samplings

    def limit_to_roi(self, roi: Polygon) -> 'Sampling2D':
        """ Computes a new Sampling2D from this one limited to its intersection with an ROI.

        :param roi: a region of interest
        :returns: a sampling limited to the X and Y coordinates which intersect the ROI.
        """
        roi_minx, roi_miny, roi_maxx, roi_maxy = roi.bounds
        x_samples = np.extract((self.x_samples >= roi_minx) & (self.x_samples <= roi_maxx),
                               self.x_samples)
        y_samples = np.extract((self.y_samples >= roi_miny) & (self.y_samples <= roi_maxy),
                               self.y_samples)
        return Sampling2D(x_samples, y_samples)

    def __str__(self) -> str:
        msg = f' N: {self.nb_samples} = {len(self._y_samples)}*{len(self._x_samples)} '
        msg += f' X[{self.upper_left_sample.x}, {self.lower_right_sample.x}] *'
        msg += f' Y[{self.upper_left_sample.y}, {self.lower_right_sample.y}]'
        return msg
