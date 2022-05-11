# -*- coding: utf-8 -*-
""" Definition of the LocalizedDataProvider base class

:author: GIROS Alain
:created: 23/06/2021
"""
from typing import Tuple, Optional  # @NoMove

from osgeo import osr, gdal

GDAL3_OR_GREATER = gdal.VersionInfo()[0] >= '3'
SWAP_COORDS_EPSG = [4326]

# TODO: get rid of PointType
PointType = Tuple[float, float]


class LocalizedDataProvider:
    """ Base class for providers which deliver data depending on some location on Earth.
    It offers the ability to store the SRS which is used by the client of the provider for
    specifying a given point on Earth as well as the methods for transforming these coordinates
    into the working SRS of the provider.
    """

    def __init__(self) -> None:

        # Default provider SRS is set to EPSG:4326
        self._provider_epsg_code = 4326

        # Default client SRS is set to EPSG:4326 as well
        self._client_epsg_code = 4326

        # Thus default coordinates transformations do nothing
        self._client_to_provider_transform: Optional[osr.CoordinateTransformation] = None
        self._provider_to_client_transform: Optional[osr.CoordinateTransformation] = None

    @property
    def client_epsg_code(self) -> int:
        """ :returns: the epsg code of the SRS which will be used in subsequent client requests
        """
        return self._client_epsg_code

    @client_epsg_code.setter
    def client_epsg_code(self, value: int) -> None:
        self._client_epsg_code = value
        self._client_to_provider_transform = None
        self._provider_to_client_transform = None

    @property
    def provider_epsg_code(self) -> int:
        """ :returns: the epsg code of the SRS used by the provider to retrieve its info
        """
        return self._provider_epsg_code

    @provider_epsg_code.setter
    def provider_epsg_code(self, value: int) -> None:
        self._provider_epsg_code = value
        self._client_to_provider_transform = None
        self._provider_to_client_transform = None

    def transform_point(self, point: PointType, altitude: float) -> Tuple[float, float, float]:
        """ Transform a point in 3D from the client SRS to the provider SRS

        :param point: (X, Y) coordinates of the point in the client SRS
        :param altitude: altitude of the point in the client SRS
        :returns: 3D coordinates in the provider SRS corresponding to the point. Meaning of
                  these coordinates depends on the provider SRS: (longitude, latitude, height) for
                  geographical SRS or (X, Y, height) for cartographic SRS.
        :warning: this method must not be called outside a worker when this class is used in a
                  dask context. This is because a osr.CoordinateTransformation object cannot be
                  serialized using pickle because it is a SwigPyObject object.
        """
        if self._client_to_provider_transform is None:
            client_srs = osr.SpatialReference()
            client_srs.ImportFromEPSG(self.client_epsg_code)
            provider_srs = osr.SpatialReference()
            provider_srs.ImportFromEPSG(self.provider_epsg_code)
            self._client_to_provider_transform = osr.CoordinateTransformation(client_srs,
                                                                              provider_srs)
        if GDAL3_OR_GREATER and self.client_epsg_code in SWAP_COORDS_EPSG:
            point = (point[1], point[0])
        transformed_point = self._client_to_provider_transform.TransformPoint(*point, altitude)
        if GDAL3_OR_GREATER and self.provider_epsg_code in SWAP_COORDS_EPSG:
            transformed_point = (transformed_point[1],
                                 transformed_point[0],
                                 transformed_point[2])
        return transformed_point

    def reverse_transform_point(self, point: PointType, altitude: float) -> \
            Tuple[float, float, float]:
        """ Transform a point in 3D from the provider SRS to the client SRS

        :param point: (X, Y) coordinates of the point in the provider SRS
        :param altitude: altitude of the point in the provider SRS
        :returns: 3D coordinates in the client SRS corresponding to the point. Meaning of
                  these coordinates depends on the client SRS: (longitude, latitude, height) for
                  geographical SRS or (X, Y, height) for cartographic SRS.
        :warning: this method must not be called outside a worker when this class is used in a
                  dask context. This is because a osr.CoordinateTransformation object cannot be
                  serialized using pickle because it is a SwigPyObject object.
        """
        if self._provider_to_client_transform is None:
            client_srs = osr.SpatialReference()
            client_srs.ImportFromEPSG(self.client_epsg_code)
            provider_srs = osr.SpatialReference()
            provider_srs.ImportFromEPSG(self.provider_epsg_code)
            self._provider_to_client_transform = osr.CoordinateTransformation(provider_srs,
                                                                              client_srs)
        if GDAL3_OR_GREATER and self.provider_epsg_code in SWAP_COORDS_EPSG:
            point = (point[1], point[0])
        transformed_point = self._provider_to_client_transform.TransformPoint(*point, altitude)
        if GDAL3_OR_GREATER and self.client_epsg_code in SWAP_COORDS_EPSG:
            transformed_point = (transformed_point[1],
                                 transformed_point[0],
                                 transformed_point[2])
        return transformed_point
