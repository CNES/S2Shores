# -*- coding: utf-8 -*-
""" Definition of the OrthoLayout class

:author: GIROS Alain
:created: 05/08/2021
"""
from typing import Optional  # @NoMove

import numpy as np  # @NoMove
from shapely.affinity import translate
from shapely.geometry import Polygon, Point

from ..generic_utils.tiling_utils import modular_sampling
from .geo_transform import GeoTransform, GdalGeoTransformType
from .image_geometry_types import MarginsType, ImageWindowType
from .sampling_2d import Sampling2D


class OrthoLayout:
    """ This class makes the link between the pixels extent of an orthorectified image and
    its projected extent in some SRS.
    """

    def __init__(self, nb_columns: int, nb_lines: int, projection: str,
                 gdal_geotransform: GdalGeoTransformType) -> None:
        """ Constructor.

        :param nb_columns: the number of columns of this image
        :param nb_lines: the number of lines of this image
        :param projection: the projection of this orthorectified image, as a wkt.
        :param gdal_geotransform: the GDAL geotransform allowing to transform cartographic
                                  coordinates into image coordinates and reciprocally.
        """
        # Extract the relevant information from one of the jp2 images
        self._nb_columns = nb_columns
        self._nb_lines = nb_lines
        self._projection = projection
        self._geo_transform = GeoTransform(gdal_geotransform)

        # Get georeferenced extent of the whole image
        self._upper_left_corner = self._geo_transform.projected_coordinates(Point(0., 0.))
        self._lower_right_corner = self._geo_transform.projected_coordinates(Point(self._nb_columns,
                                                                                   self._nb_lines))

    @property
    def epsg_code(self) -> int:
        """ :returns: the epsg code of the projection
        """
        return int(self._projection.split(',')[-1][1:-3])

    # TODO: define steps default values based on resolution
    def get_samples_positions(self, step_x: float, step_y: float, local_margins: MarginsType,
                              roi: Optional[Polygon] = None) -> Sampling2D:
        """ x_samples, y_samples are the coordinates  of the final samples in georeferenced system
        sampled from a starting position with different steps on X and Y axis.

        :param step_x: the sampling step to use along the X axis to sample this image
        :param step_y: the sampling step to use along the Y axis to sample this image
        :param local_margins: the margins to consider around the samples
        :param roi: a rectangle describing the ROI if any.
        :returns: the chosen samples specified by the cross product of X samples and Y samples
        """
        # Compute all the sampling X and Y coordinates falling inside the image domain
        left_sample_index, right_sample_index = modular_sampling(self._upper_left_corner.x,
                                                                 self._lower_right_corner.x,
                                                                 step_x)

        bottom_sample_index, top_sample_index = modular_sampling(self._lower_right_corner.y,
                                                                 self._upper_left_corner.y,
                                                                 step_y)
        x_samples = np.arange(left_sample_index, right_sample_index + 1) * step_x
        y_samples = np.arange(bottom_sample_index, top_sample_index + 1) * step_y

        # Adding half resolution to point at the pixel center
        x_samples += self._geo_transform.x_resolution / 2.
        y_samples += self._geo_transform.y_resolution / 2.

        sampling = Sampling2D(x_samples, y_samples)
        return self._get_acceptable_samples(sampling, local_margins, roi)

    def _get_acceptable_samples(self, sampling: Sampling2D, local_margins: MarginsType,
                                roi: Optional[Polygon] = None) -> Sampling2D:
        """ Filter out the samples which does not fall inside the ROI is it is defined and whose
        window centered on them does not belong to the image footprint.

        :param sampling: the cartographic coordinates of the samples to filter along the X axis
        :param local_margins: the margins to consider around the samples
        :param roi: a rectangle describing the ROI if any.
        :returns: the chosen samples specified by the cross product of X samples and Y samples
        """
        if roi is not None:
            sampling = sampling.limit_to_roi(roi)

        x_samples = sampling.x_samples
        y_samples = sampling.y_samples
        acceptable_samples_x = []
        for x_coord in x_samples:
            for y_coord in y_samples:
                point = Point(x_coord, y_coord)
                if roi is None or roi.contains(point):
                    if self.is_window_inside(point, local_margins):
                        acceptable_samples_x.append(x_coord)
                        break

        acceptable_samples_y = []
        for y_coord in y_samples:
            for x_coord in acceptable_samples_x:
                point = Point(x_coord, y_coord)
                if roi is None or roi.contains(point):
                    if self.is_window_inside(point, local_margins):
                        acceptable_samples_y.append(y_coord)
                        break

        x_samples = np.array(acceptable_samples_x)
        y_samples = np.array(acceptable_samples_y)

        return Sampling2D(x_samples, y_samples)

    def is_window_inside(self, point: Point, margins: MarginsType) -> bool:
        """ Determine if a window centered on a given point is fully inside the OrthoLayout

        :param point: the center point
        :param margins: the margins defining the window around the point
        :returns: True if the window is fully inside the layout, False otherwise
        """
        line_start, line_stop, col_start, col_stop = self.window_pixels(point, margins)
        return (line_start >= 0 and line_stop < self._nb_lines and
                col_start >= 0 and col_stop < self._nb_columns)

    def window_pixels(self, point: Point, margins: MarginsType,
                      line_start: int = 0, col_start: int = 0) -> ImageWindowType:
        """ Given a point defined in the projected domain, computes a rectangle of pixels centered
        on the pixel containing this point and taking into account the specified margins.
        No check is done at this level to verify that the rectangle is contained within the pixels
        space.

        :param point: the center point
        :param margins: the margins to consider around the point in order to build the window.
        :param line_start: line number in the image from which the window coordinates are computed
        :param col_start: column number in the image from which the window coordinates are computed
        :returns: the window as a tuple of four coordinates relative to line_start and col_start:
                  - start and stop lines (both included) in the image space defining the window
                  - start and stop columns  (both included) in the image space defining the window
        """
        # define the sub window domain in projected coordinates
        upper_left_corner = translate(point, xoff=-margins[0], yoff=margins[3])
        lower_right_corner = translate(point, xoff=margins[1], yoff=-margins[2])

        # compute the sub window domain in pixels
        image_upper_left_corner = self._geo_transform.image_coordinates(upper_left_corner)
        image_lower_right_corner = self._geo_transform.image_coordinates(lower_right_corner)
        window_col_start, window_line_start = image_upper_left_corner.coords[0]
        window_col_stop, window_line_stop = image_lower_right_corner.coords[0]
        window_pix = (int(window_line_start) - line_start,
                      int(window_line_stop) - line_start,
                      int(window_col_start) - col_start,
                      int(window_col_stop) - col_start)
        return window_pix
