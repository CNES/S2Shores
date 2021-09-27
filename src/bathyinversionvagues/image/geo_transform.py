# -*- coding: utf-8 -*-
""" Definition of the GeoTransform class

:author: GIROS Alain
:created: 26/04/2021
"""
from .image_geometry_types import PointType, GdalGeoTransformType


class GeoTransform:
    """ Definition of a geotransform as implemented in gdal, allowing to transform cartographic
    coordinates into image coordinates in both directions:

    - (X, Y) -> (C, L)
    - (C, L) -> (X, Y)

    According to this model the following holds:

    - the pixel at the upper left corner of the image, is indexed by C=0, L=0
    - the cartographic coordinates of a pixel are those of the upper left corner of that pixel
    - resolution in the Y direction is generally negative to account for the opposite directions
      of the Y axis and the line numbers
    """

    def __init__(self, geo_transform: GdalGeoTransformType) -> None:
        """ Create a GeoTransform instance from the provided parameters

        :param geo_transform: A sequence of 6 floats specifying the geo transform from pixel
                              coordinates to cartographic coordinates with the following meaning:
                              - geo_transform[0] : X coordinate of the origin
                              - geo_transform[1] : resolution of a pixel along the X direction
                              - geo_transform[2] : rotation between image and cartographic coords
                              - geo_transform[3] : Y coordinate of the origin
                              - geo_transform[4] : rotation between image and cartographic coords
                              - geo_transform[5] : resolution of a pixel along the Y direction
        """
        self.geo_transform = geo_transform

    @property
    def x_resolution(self) -> float:
        """ :returns: the resolution of the image along the X axis in the units of the geo transform
        assuming that X and Y resolutions are identical
        """
        return self.geo_transform[1]

    @property
    def y_resolution(self) -> float:
        """ :returns: the resolution of the image along the Y axis in the units of the geo transform
        """
        return self.geo_transform[5]

    @property
    def resolution(self) -> float:
        """ :returns: the resolution of the image in the units of the geo transform,
        assuming that X and Y resolutions are identical
        """
        return self.x_resolution

    @property
    def x_y_resolutions_equal(self) -> bool:
        """ :returns: True if the absolute values of X and Y resolutions are equal
        """
        return self.x_resolution == -self.y_resolution

    def projected_coordinates(self, point_column: float, point_line: float) -> PointType:
        """ Computes the georeferenced coordinates of a point defined by its coordinates
        in the image

        :param point_column: the point column coordinate, possibly non integer
        :param point_line: the point line coordinate, possibly non integer
        :returns: the x and y point coordinates in the projection system associated to this image.
        """
        projected_x = (self.geo_transform[0] +
                       point_column * self.geo_transform[1] + point_line * self.geo_transform[2])
        projected_y = (self.geo_transform[3] +
                       point_column * self.geo_transform[4] + point_line * self.geo_transform[5])
        return projected_x, projected_y

    def image_coordinates(self, projected_x: float, projected_y: float) -> PointType:
        """ Computes the images coordinates of a point defined by its coordinates in the projection
        associated to this geotransform.

        :param projected_x: the point coordinate along the X projection axis
        :param projected_y: the point coordinate along the Y projection axis
        :returns: the corresponding column and line coordinates in the image.
        """
        det = (self.geo_transform[1] * self.geo_transform[5] -
               self.geo_transform[2] * self.geo_transform[4])
        offset_column = (self.geo_transform[2] * self.geo_transform[3] -
                         self.geo_transform[0] * self.geo_transform[5])
        offset_line = (self.geo_transform[0] * self.geo_transform[4] -
                       self.geo_transform[1] * self.geo_transform[3])

        point_line = (self.geo_transform[1] * projected_y -
                      self.geo_transform[4] * projected_x + offset_line) / det
        point_column = (self.geo_transform[5] * projected_x -
                        self.geo_transform[2] * projected_y + offset_column) / det
        return point_column, point_line
