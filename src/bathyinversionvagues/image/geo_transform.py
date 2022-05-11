# -*- coding: utf-8 -*-
""" Definition of the GeoTransform class

:author: GIROS Alain
:created: 26/04/2021
"""
from typing import Sequence

from shapely.geometry.point import Point


GdalGeoTransformType = Sequence[float]


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

    def projected_coordinates(self, image_point: Point) -> Point:
        """ Computes the georeferenced coordinates of a point defined by its coordinates
        in the image

        :param image_point: the point in image coordinates, possibly non integer
        :returns: the point in the projection system associated to this image.
        """
        # TODO: use shapely.affinity
        projected_x = (self.geo_transform[0] + image_point.x * self.geo_transform[1] +
                       image_point.y * self.geo_transform[2])
        projected_y = (self.geo_transform[3] + image_point.x * self.geo_transform[4] +
                       image_point.y * self.geo_transform[5])
        return Point(projected_x, projected_y)

    def image_coordinates(self, projected_point: Point) -> Point:
        """ Computes the images coordinates of a point defined in the projection associated to this
        geotransform.

        :param projected_point: the point in projection coordinates
        :returns: the corresponding point in image coordinates.
        """
        det = (self.geo_transform[1] * self.geo_transform[5] -
               self.geo_transform[2] * self.geo_transform[4])
        offset_column = (self.geo_transform[2] * self.geo_transform[3] -
                         self.geo_transform[0] * self.geo_transform[5])
        offset_line = (self.geo_transform[0] * self.geo_transform[4] -
                       self.geo_transform[1] * self.geo_transform[3])

        point_line = (self.geo_transform[1] * projected_point.y -
                      self.geo_transform[4] * projected_point.x + offset_line) / det
        point_column = (self.geo_transform[5] * projected_point.x -
                        self.geo_transform[2] * projected_point.y + offset_column) / det
        return Point(point_column, point_line)
