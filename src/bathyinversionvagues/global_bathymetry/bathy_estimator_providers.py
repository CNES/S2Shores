# -*- coding: utf-8 -*-
""" Definition of the BathyEstimator abstract class

:author: GIROS Alain
:created: 17/05/2021
"""
from pathlib import Path
from typing import Optional, Union  # @NoMove

from shapely.geometry import Point


from ..data_providers.delta_time_provider import DeltaTimeProvider, NoDeltaTimeProviderError
from ..data_providers.dis_to_shore_provider import (InfinityDisToShoreProvider, DisToShoreProvider,
                                                    NetCDFDisToShoreProvider, GeotiffDisToShoreProvider)
from ..data_providers.gravity_provider import (LatitudeVaryingGravityProvider, GravityProvider,
                                               ConstantGravityProvider)
from ..data_providers.roi_provider import (RoiProvider, VectorFileRoiProvider)
from ..image.ortho_stack import OrthoStack


class BathyEstimatorProviders:
    """ Management of bathymetry computation and parameters on a single product. Computation
    is split in several cartographic tiles, which must be run separately, either in parallel or
    sequentially.
    """

    def __init__(self, ortho_stack: OrthoStack) -> None:
        """Create a BathyEstimator object and set necessary informations

        :param ortho_stack: the orthorectified stack onto which bathymetry must be estimated.
        """
        self._ortho_stack = ortho_stack
        self._distoshore_provider: DisToShoreProvider
        # set InfinityDisToShoreProvider as default DisToShoreProvider
        self.set_distoshore_provider(provider_info=InfinityDisToShoreProvider())

        self._gravity_provider: GravityProvider
        # set LatitudeVaryingGravityProvider as default GravityProvider
        self.set_gravity_provider(provider_info=LatitudeVaryingGravityProvider())

        # No default DeltaTimeProvider
        self._delta_time_provider: Optional[DeltaTimeProvider] = None

        # No default RoiProvider
        self._roi_provider: Optional[RoiProvider] = None
        self._limit_to_roi = False

    def set_distoshore_provider(
            self, provider_info: Optional[Union[Path, DisToShoreProvider]] = None) -> None:
        """ Sets the DisToShoreProvider to use with this estimator

        :param provider_info: Either the DisToShoreProvider to use or a path to a netCDF or Geotiff file
                           assuming a geographic NetCDF or Geotiff format.
        """
        if isinstance(provider_info, DisToShoreProvider):
            distoshore_provider = provider_info
        elif isinstance(provider_info, Path):
            if (Path(provider_info).suffix.lower() == '.nc'):
                distoshore_provider = NetCDFDisToShoreProvider(provider_info, 4326,
                                                               x_axis_label='lon',
                                                               y_axis_label='lat')
            elif (Path(provider_info).suffix.lower() == '.tif'):
                distoshore_provider = GeotiffDisToShoreProvider(provider_info)
        else:
            # None or some other type, keep the current provider
            distoshore_provider = self._distoshore_provider

        # Set private attribute.
        self._distoshore_provider = distoshore_provider
        if self._distoshore_provider is not None:
            self._distoshore_provider.client_epsg_code = self._ortho_stack.epsg_code

    def get_distoshore(self, point: Point) -> float:
        """ Provides the distance from a given point to the nearest shore.

        :param point: the point from which the distance to shore is requested.
        :returns: the distance from the point to the nearest shore (km).
        """
        return self._distoshore_provider.get_distoshore(point)

    def set_roi_provider(self, provider_info: Optional[Union[Path, RoiProvider]] = None,
                         limit_to_roi: bool = False) -> None:
        """ Sets the RoiProvider to use with this estimator

        :param provider_info: Either the RoiProvider to use or a path to a vector file containing
                              the ROI or None if no provider change.
        :param limit_to_roi: if True, the produced bathymetry will be limited to a bounding box
                             enclosing the Roi with some margins.
        """
        roi_provider: Optional[RoiProvider]
        if isinstance(provider_info, RoiProvider):
            roi_provider = provider_info
        elif isinstance(provider_info, Path):
            roi_provider = VectorFileRoiProvider(provider_info)
        else:
            # None or some other type, keep the current provider
            roi_provider = self._roi_provider

        # Set private attribute.
        self._roi_provider = roi_provider
        if self._roi_provider is not None:
            self._roi_provider.client_epsg_code = self._ortho_stack.epsg_code
            self._limit_to_roi = limit_to_roi

    def is_inside_roi(self, point: Point) -> bool:
        """ Test if a point is inside the ROI

        :param point: the point to test
        :returns: True if the point lies inside the ROI.
        """
        if self._roi_provider is None:
            return True
        return self._roi_provider.contains(point)

    def set_gravity_provider(self,
                             provider_info: Optional[Union[str, GravityProvider]] = None) -> None:
        """ Sets the GravityProvider to use with this estimator .

        :param provider_info: an instance of GravityProvider or the name of a well known gravity
                              provider to use. If None the current provider is left unchanged.
        :raises ValueError: when the gravity provider name is unknown
        """
        if isinstance(provider_info, GravityProvider):
            gravity_provider = provider_info
        elif isinstance(provider_info, str):
            if provider_info.upper() not in ['CONSTANT', 'LATITUDE_VARYING']:
                raise ValueError('Gravity provider type unknown : ', provider_info)
            # No need to set LatitudeVaryingGravityProvider as it is the BathyEstimator default.
            if provider_info.upper() == 'CONSTANT':
                gravity_provider = ConstantGravityProvider()
            else:
                gravity_provider = LatitudeVaryingGravityProvider()
        else:
            # None or some other type, keep the current provider
            gravity_provider = self._gravity_provider

        # Set private attribute.
        self._gravity_provider = gravity_provider
        if self._gravity_provider is not None:
            self._gravity_provider.client_epsg_code = self._ortho_stack.epsg_code

    def get_gravity(self, point: Point, altitude: float = 0.) -> float:
        """ Returns the gravity at some point expressed by its X, Y and H coordinates in some SRS,
        using the gravity provider associated to this bathymetry estimator.

        :param point: a point expressed in the SRS coordinates set for this provider
        :param altitude: the altitude of the point in the SRS set for this provider
        :returns: the acceleration due to gravity at this point (m/s2).
        """
        return self._gravity_provider.get_gravity(point, altitude)

    def set_delta_time_provider(
            self, provider_info: Optional[Union[Path, DeltaTimeProvider]] = None) -> None:
        """ Sets the DeltaTimeProvider to use with this estimator.

        :param provider_info: Either the DeltaTimeProvider to use or a path to a file or a
                              directory to ba used by the associated OrthoStack to build its
                              provider, or None to leave the provider unchanged.
        """
        delta_time_provider: Optional[DeltaTimeProvider]
        if isinstance(provider_info, DeltaTimeProvider):
            delta_time_provider = provider_info
        else:
            delta_time_provider = self._ortho_stack.create_delta_time_provider(provider_info)

        # Set private attribute.
        self._delta_time_provider = delta_time_provider
        if self._delta_time_provider is not None:
            self._delta_time_provider.client_epsg_code = self._ortho_stack.epsg_code

    @property
    def delta_time_provider(self) -> DeltaTimeProvider:
        """ :returns: The delta time provider associated to this estimator.
        :raises NoDeltaTimeProviderError: when no provider is defined.
        """
        if self._delta_time_provider is None:
            raise NoDeltaTimeProviderError()
        return self._delta_time_provider
