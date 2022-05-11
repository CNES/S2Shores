# -*- coding: utf-8 -*-
""" Definition of the RoiProvider classes

:author: GIROS Alain
:created: 07/12/2021
"""
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, cast  # @NoMove

from osgeo import ogr
from shapely.geometry import Polygon, Point, MultiPolygon, box

from .localized_data_provider import LocalizedDataProvider


class RoiProvider(ABC, LocalizedDataProvider):
    """ A Roi provider is a service which is able to test if a point specified by its coordinates
    in some SRS in inside a Region Of Interest expressed as a set of polygons defined in another
    SRS.
    """

    @abstractmethod
    def contains(self, point: Point) -> bool:
        """ Test if a point is inside the ROI

        :param point: the point in the SRS coordinates set for this provider
        :returns: True if the point lies inside the ROI
        """

    @abstractmethod
    def bounding_box(self, margin: float = 0.1) -> Polygon:
        """ Compute the bounding box enclosing the ROI expressed in the client SRS with some margins

        :param margin: a margin to apply to the smallest bounding box expressed as a percentage of
                       its width and height.
        :returns: the bounding box
        """


class VectorFileRoiProvider(RoiProvider):
    """ A RoiProvider where the ROI is defined by a vector file in some standard format.
    """

    def __init__(self, vector_file_path: Path) -> None:
        """ Create a NetCDFDisToShoreProvider object and set necessary informations

        :param vector_file_path: full path of a vector file containing the ROI as a non empty set of
                                 polygons
        """
        super().__init__()

        self._polygons: Optional[MultiPolygon] = None
        self._vector_file_path = vector_file_path

    def bounding_box(self, margin: float = 0.1) -> Polygon:
        if self._polygons is None:
            self._load_polygons()
        x_min, y_min, x_max, y_max = cast(MultiPolygon, self._polygons).bounds
        x_min_client, y_min_client, _ = self.reverse_transform_point((x_min, y_min), 0.)
        x_max_client, y_max_client, _ = self.reverse_transform_point((x_max, y_max), 0.)
        delta_width = (x_max_client - x_min_client) * margin / 2.
        delta_height = (y_max_client - y_min_client) * margin / 2.
        bouding_box_polygon = box(x_min_client - delta_width, y_max_client + delta_height,
                                  x_max_client + delta_width, y_min_client - delta_height)
        self._provider_to_client_transform = None  # provision for avoiding core dumps with ogr.
        return bouding_box_polygon

    def contains(self, point: Point) -> bool:
        if self._polygons is None:
            self._load_polygons()
        tranformed_point = Point(*self.transform_point((point.x, point.y), 0.))
        return cast(MultiPolygon, self._polygons).contains(tranformed_point)

    def _load_polygons(self) -> None:
        """ Read the vector file and loads the polygons contained in its first layer
        """
        polygons = []
        dataset = ogr.Open(str(self._vector_file_path))
        layer = dataset.GetLayerByIndex(0)
        self.provider_epsg_code = int(layer.GetSpatialRef().GetAuthorityCode(None))
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            polygon_ring = feature.GetGeometryRef().GetGeometryRef(0)

            # We use shapely Polygon in order to circumvent a core dump when using OGR with dask
            polygon = Polygon(polygon_ring.GetPoints())
            polygons.append(polygon)
        self._polygons = MultiPolygon(polygons)
