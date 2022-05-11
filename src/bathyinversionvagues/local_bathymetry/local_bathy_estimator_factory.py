# -*- coding: utf-8 -*-
""" Selection of the desired local bathymetry estimator

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""
from typing import Dict, Optional, Type, TYPE_CHECKING  # @NoMove

from shapely.geometry import Point
import numpy as np

from ..bathy_debug.spatial_dft_bathy_estimator_debug import SpatialDFTBathyEstimatorDebug
from ..bathy_debug.temporal_correlation_bathy_estimator_debug import \
    TemporalCorrelationBathyEstimatorDebug
from ..image.ortho_sequence import OrthoSequence

from .local_bathy_estimator import LocalBathyEstimator
from .spatial_correlation_bathy_estimator import SpatialCorrelationBathyEstimator
from .spatial_dft_bathy_estimator import SpatialDFTBathyEstimator
from .temporal_correlation_bathy_estimator import TemporalCorrelationBathyEstimator


if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport

# Dictionary of classes to be instanciated for each local bathymetry estimator
LOCAL_BATHY_ESTIMATION_CLS: Dict[str, Type[LocalBathyEstimator]]
LOCAL_BATHY_ESTIMATION_CLS_DEBUG: Dict[str, Type[LocalBathyEstimator]]
LOCAL_BATHY_ESTIMATION_CLS = {'SPATIAL_DFT': SpatialDFTBathyEstimator,
                              'TEMPORAL_CORRELATION': TemporalCorrelationBathyEstimator,
                              'SPATIAL_CORRELATION': SpatialCorrelationBathyEstimator}

LOCAL_BATHY_ESTIMATION_CLS_DEBUG = {'SPATIAL_DFT': SpatialDFTBathyEstimatorDebug,
                                    'TEMPORAL_CORRELATION': TemporalCorrelationBathyEstimatorDebug,
                                    'SPATIAL_CORRELATION': SpatialCorrelationBathyEstimator}


def local_bathy_estimator_factory(location: Point, ortho_sequence: OrthoSequence,
                                  global_estimator: 'BathyEstimator',
                                  selected_directions: Optional[np.ndarray] = None) \
        -> LocalBathyEstimator:
    """ Build an instance of a local bathymetry estimator from its code, with potential debug
    capabilities.

    :returns: an instance of a local bathymetry estimator suitable for running estimation
    """
    local_bathy_estimator_cls = get_local_bathy_estimator_cls(global_estimator.local_estimator_code,
                                                              global_estimator.debug_sample)
    return local_bathy_estimator_cls(location, ortho_sequence, global_estimator,
                                     selected_directions)


def get_local_bathy_estimator_cls(local_estimator_code: str,
                                  debug_mode: bool) -> Type[LocalBathyEstimator]:
    """ return the local bathymetry estimator class corresponding to a given estimator code

    :returns: the local bathymetry estimator class corresponding to a given estimator code
    :raises NotImplementedError: when the requested bathymetry estimator is unknown
    """
    try:
        if debug_mode:
            local_bathy_estimator_cls = LOCAL_BATHY_ESTIMATION_CLS_DEBUG[local_estimator_code]
        else:
            local_bathy_estimator_cls = LOCAL_BATHY_ESTIMATION_CLS[local_estimator_code]
    except KeyError:
        msg = f'{local_estimator_code} is not a supported local bathymetry estimation method'
        if debug_mode:
            msg += f' with debug mode'
        raise NotImplementedError(msg)
    return local_bathy_estimator_cls
