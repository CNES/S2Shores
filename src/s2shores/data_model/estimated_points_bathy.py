# -*- coding: utf-8 -*-
""" Definition of the EstimatedBathy class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:created: 02/03/2023
"""
from datetime import datetime
from typing import Any, Dict, Hashable, List, Mapping, Tuple, Union

import numpy as np  # @NoMove
from xarray import DataArray  # @NoMove

from ..data_model.estimated_bathy import (DEBUG_LAYER, EXPERT_LAYER, METERS_UNIT, NOMINAL_LAYER,
                                          SPATIAL_REF, EstimatedBathy)
from ..waves_exceptions import WavesEstimationAttributeError
from .bathymetry_sample_estimations import BathymetrySampleEstimations

DIMS_INDEX_NKEEP_TIME = ['time', 'kKeep', 'index']
DIMS_INDEX_TIME = ['time', 'index']

# Provides a mapping from entries into the output dictionary of a local estimator to a netCDF layer.
BATHY_PRODUCT_DEF: Dict[str, Dict[str, Any]] = {
    'status': {'layer_type': NOMINAL_LAYER,
               'layer_name': 'Status',
               'dimensions': DIMS_INDEX_TIME,
               'data_type': np.ushort,
               'fill_value': 0,
               'precision': 0,
               'attrs': {'Dimension': 'Flags',
                         'long_name': 'Bathymetry estimation status',
                         'comment': '0: SUCCESS, 1: FAIL, 2: ON_GROUND, '
                                    '3: NO_DATA, 4: NO_DELTA_TIME , '
                                    '5: OUTSIDE_ROI, 6: BEYOND_OFFSHORE_LIMIT'}},
    'status_debug': {'layer_type': DEBUG_LAYER,
               'layer_name': 'Status_Debug',
               'dimensions': DIMS_INDEX_NKEEP_TIME,
               'data_type': np.ushort,
               'fill_value': 0,
               'precision': 0,
               'attrs': {'Dimension': 'Flags',
                         'long_name': 'Bathymetry estimation status per kKeep',
                         'comment': '0: SUCCESS, 1: FAIL, 2: ON_GROUND, '
                                    '3: NO_DATA, 4: NO_DELTA_TIME , '
                                    '5: OUTSIDE_ROI, 6: BEYOND_OFFSHORE_LIMIT'}},                                    
    'depth': {'layer_type': NOMINAL_LAYER,
              'layer_name': 'Depth',
              'dimensions': DIMS_INDEX_NKEEP_TIME,
              'data_type': np.float32,
              'fill_value': np.nan,
              'precision': 2,
              'attrs': {'Dimension': METERS_UNIT,
                        'long_name': 'Raw estimated depth',
                        'grid_mapping': SPATIAL_REF,
                        'coordinates': SPATIAL_REF}},
    'direction_from_north': {'layer_type': NOMINAL_LAYER,
                             'layer_name': 'Direction',
                             'dimensions': DIMS_INDEX_NKEEP_TIME,
                             'data_type': np.float32,
                             'fill_value': np.nan,
                             'precision': 1,
                             'attrs': {'Dimension': 'degree',
                                       'long_name': 'Wave direction',
                                       'comment': 'Direction from North',
                                       'grid_mapping': SPATIAL_REF,
                                       'coordinates': SPATIAL_REF}},
    'celerity': {'layer_type': NOMINAL_LAYER,
                 'layer_name': 'Celerity',
                 'dimensions': DIMS_INDEX_NKEEP_TIME,
                 'data_type': np.float32,
                 'fill_value': np.nan,
                 'precision': 2,
                 'attrs': {'Dimension': 'Meters per second [m/sec]',
                           'long_name': 'Wave celerity',
                           'grid_mapping': SPATIAL_REF,
                           'coordinates': SPATIAL_REF}},
    'absolute_delta_position': {'layer_type': EXPERT_LAYER,
                                'layer_name': 'Propagated distance',
                                'dimensions': DIMS_INDEX_NKEEP_TIME,
                                'data_type': np.float32,
                                'fill_value': np.nan,
                                'precision': 2,
                                'attrs': {'Dimension': METERS_UNIT,
                                          'long_name': 'Distance used for measuring wave celerity',
                                          'comment': 'The actual sign of this quantity equals the '
                                          'sign of Delta Acquisition Time',
                                          'grid_mapping': SPATIAL_REF,
                                          'coordinates': SPATIAL_REF}},
    'wavelength': {'layer_type': NOMINAL_LAYER,
                   'layer_name': 'Wavelength',
                   'dimensions': DIMS_INDEX_NKEEP_TIME,
                   'data_type': np.float32,
                   'fill_value': np.nan,
                   'precision': 1,
                   'attrs': {'Dimension': METERS_UNIT,
                             'long_name': 'Wavelength',
                             'grid_mapping': SPATIAL_REF,
                             'coordinates': SPATIAL_REF}},
    'wavenumber': {'layer_type': EXPERT_LAYER,
                   'layer_name': 'Wavenumber',
                   'dimensions': DIMS_INDEX_NKEEP_TIME,
                   'data_type': np.float32,
                   'fill_value': np.nan,
                   'precision': 5,
                   'attrs': {'Dimension': 'Per Meter [m-1]',
                             'long_name': 'Wavenumber',
                             'grid_mapping': SPATIAL_REF,
                             'coordinates': SPATIAL_REF}},
    'period': {'layer_type': EXPERT_LAYER,
               'layer_name': 'Period',
               'dimensions': DIMS_INDEX_NKEEP_TIME,
               'data_type': np.float32,
               'fill_value': np.nan,
               'precision': 2,
               'attrs': {'Dimension': 'Seconds [sec]',
                         'long_name': 'Wave period',
                         'grid_mapping': SPATIAL_REF,
                         'coordinates': SPATIAL_REF}},
    'distance_to_shore': {'layer_type': EXPERT_LAYER,
                          'layer_name': 'Distoshore',
                          'dimensions': DIMS_INDEX_TIME,
                          'data_type': np.float32,
                          'fill_value': np.nan,
                          'precision': 3,
                          'attrs': {'Dimension': 'Kilometers [km]',
                                    'long_name': 'Distance to the shore',
                                    'grid_mapping': SPATIAL_REF,
                                    'coordinates': SPATIAL_REF}},
    'delta_celerity': {'layer_type': EXPERT_LAYER,
                       'layer_name': 'Delta Celerity',
                       'dimensions': DIMS_INDEX_NKEEP_TIME,
                       'data_type': np.float32,
                       'fill_value': np.nan,
                       'precision': 2,
                       'attrs': {'Dimension': 'Meters per seconds2 [m/sec2]',
                                 'long_name': 'delta celerity',
                                 'grid_mapping': SPATIAL_REF,
                                 'coordinates': SPATIAL_REF}},
    'absolute_delta_phase': {'layer_type': EXPERT_LAYER,
                             'layer_name': 'Delta Phase',
                             'dimensions': DIMS_INDEX_NKEEP_TIME,
                             'data_type': np.float32,
                             'fill_value': np.nan,
                             'precision': 8,
                             'attrs': {'Dimension': 'Radians [rd]',
                                       'long_name': 'Delta Phase',
                                       'comment': 'The actual sign of this quantity equals the '
                                       'sign of Delta Acquisition Time',
                                       'grid_mapping': SPATIAL_REF,
                                       'coordinates': SPATIAL_REF}},
    'gravity': {'layer_type': EXPERT_LAYER,
                'layer_name': 'Gravity',
                'dimensions': DIMS_INDEX_TIME,
                'data_type': np.float32,
                'fill_value': np.nan,
                'precision': 4,
                'attrs': {'Dimension': 'Acceleration [m/s2]',
                          'long_name': 'Gravity',
                          'grid_mapping': SPATIAL_REF,
                          'coordinates': SPATIAL_REF}},
    'delta_time': {'layer_type': EXPERT_LAYER,
                   'layer_name': 'Delta Acquisition Time',
                   'dimensions': DIMS_INDEX_TIME,
                   'data_type': np.float32,
                   'fill_value': np.nan,
                   'precision': 4,
                   'attrs': {'Dimension': 'Duration (s)',
                             'long_name': 'Delta Time',
                             'comment': 'The time length of the sequence of images used for '
                                        'estimation. May be positive or negative to account for '
                                        'the chronology of start and stop images',
                                        'grid_mapping': SPATIAL_REF,
                                        'coordinates': SPATIAL_REF}},
    'absolute_stroboscopic_factor': {'layer_type': EXPERT_LAYER,
                                     'layer_name': 'Stroboscopic Factor',
                                     'dimensions': DIMS_INDEX_NKEEP_TIME,
                                     'data_type': np.float32,
                                     'fill_value': np.nan,
                                     'precision': 4,
                                     'attrs': {'Dimension': 'Unitless',
                                               'long_name': '|delta_time| / period',
                                               'grid_mapping': SPATIAL_REF,
                                               'coordinates': SPATIAL_REF}},
    'absolute_stroboscopic_factor_offshore': {'layer_type': EXPERT_LAYER,
                                              'layer_name': 'Stroboscopic Factor Offshore',
                                              'dimensions': DIMS_INDEX_NKEEP_TIME,
                                              'data_type': np.float32,
                                              'fill_value': np.nan,
                                              'precision': 4,
                                              'attrs': {'Dimension': 'Unitless',
                                                        'long_name': '|delta_time| / '
                                                                     'period_offshore',
                                                                     'grid_mapping': SPATIAL_REF,
                                                                     'coordinates': SPATIAL_REF}},
    'linearity': {'layer_type': EXPERT_LAYER,
                  'layer_name': 'Wave Linearity',
                  'dimensions': DIMS_INDEX_NKEEP_TIME,
                  'data_type': np.float32,
                  'fill_value': np.nan,
                  'precision': 3,
                  'attrs': {'Dimension': 'Unitless',
                            'long_name': 'linearity coefficient: c^2k/g',
                            'comment': 'Linear dispersion relation limits: transition from '
                                       'shallow water to intermediate water around 0.3, transition '
                                       'from intermediate water to deep water around 0.9',
                                       'grid_mapping': SPATIAL_REF,
                                       'coordinates': SPATIAL_REF}},
    'period_offshore': {'layer_type': EXPERT_LAYER,
                        'layer_name': 'Period Offshore',
                        'dimensions': DIMS_INDEX_NKEEP_TIME,
                        'data_type': np.float32,
                        'fill_value': np.nan,
                        'precision': 2,
                        'attrs': {'Dimension': 'Seconds [sec]',
                                  'long_name': 'Period of the wave field if it was offshore',
                                  'grid_mapping': SPATIAL_REF,
                                  'coordinates': SPATIAL_REF}},
    'energy': {'layer_type': DEBUG_LAYER,
               'layer_name': 'Energy',
               'dimensions': DIMS_INDEX_NKEEP_TIME,
               'data_type': np.float32,
               'fill_value': np.nan,
               'precision': 1,
               'attrs': {'Dimension': 'Joules per Meter2 [J/m2]',
                         'long_name': 'Energy',
                         'grid_mapping': SPATIAL_REF,
                         'coordinates': SPATIAL_REF}},
    'relative_period': {'layer_type': DEBUG_LAYER,
                        'layer_name': 'Relative Period',
                        'dimensions': DIMS_INDEX_NKEEP_TIME,
                        'data_type': np.float32,
                        'fill_value': np.nan,
                        'precision': 3,
                        'attrs': {'Dimension': 'Unitless',
                                  'long_name': 'period_offshore / period',
                                  'comment': 'Also known as the dimensionless period',
                                  'grid_mapping': SPATIAL_REF,
                                  'coordinates': SPATIAL_REF}},
    'relative_wavelength': {'layer_type': DEBUG_LAYER,
                            'layer_name': 'Relative Wavelength',
                            'dimensions': DIMS_INDEX_NKEEP_TIME,
                            'data_type': np.float32,
                            'fill_value': np.nan,
                            'precision': 3,
                            'attrs': {'Dimension': 'Unitless',
                                      'long_name': 'wavelength_offshore / wavelength',
                                      'comment': 'Also known as the dimensionless wavelength',
                                      'grid_mapping': SPATIAL_REF,
                                      'coordinates': SPATIAL_REF}},
    'energy_ratio': {'layer_type': DEBUG_LAYER,
                     'layer_name': 'Energy Ratio',
                     'dimensions': DIMS_INDEX_NKEEP_TIME,
                     'data_type': np.float32,
                     'fill_value': np.nan,
                     'precision': 3,
                     'attrs': {'Dimension': 'Joules per Meter2 [J/m2]',
                               'long_name': 'energy_ratio',
                               'grid_mapping': SPATIAL_REF,
                               'coordinates': SPATIAL_REF}},
    'x': {'layer_type': NOMINAL_LAYER,
          'layer_name': 'X',
          'dimensions': DIMS_INDEX_TIME,
          'data_type': np.float32,
          'fill_value': np.nan,
          'precision': 3,
          'attrs': {'Dimension': 'UTM',
                    'long_name': 'x_coordinates',
                    'grid_mapping': SPATIAL_REF,
                    'coordinates': SPATIAL_REF}},
    'y': {'layer_type': NOMINAL_LAYER,
          'layer_name': 'Y',
          'dimensions': DIMS_INDEX_TIME,
          'data_type': np.float32,
          'fill_value': np.nan,
          'precision': 3,
          'attrs': {'Dimension': 'UTM',
                    'long_name': 'y_coordinates',
                    'grid_mapping': SPATIAL_REF,
                    'coordinates': SPATIAL_REF}},
}


class EstimatedPointsBathy(EstimatedBathy):
    """ This class gathers all the estimated bathymetry samples in a whole dataset.
    """

    bathy_product_def = BATHY_PRODUCT_DEF

    def __init__(self, nb_samples, acq_time: str) -> None:
        """ Define dimensions for which the estimated bathymetry samples will be defined.

        :param nb_samples: the number of samples of the estimated bathymetry
        :param acq_time: the time at which the bathymetry samples are estimated
        """
        super().__init__(acq_time)
        # data is stored as a 1D array of python objects, here a dictionary containing bathy fields.
        self.estimated_bathy = np.empty((nb_samples), dtype=np.object_)

    def store_estimations(self, index: int, bathy_estimations: BathymetrySampleEstimations) -> None:
        """ Store a set of bathymetry estimations at some location """
        self.estimated_bathy[index] = bathy_estimations

    def _build_data_array(self, sample_property: str,
                          layer_definition: Dict[str, Any], nb_keep: int) -> DataArray:
        """ Build an xarray DataArray containing one estimated bathymetry property.

        :param sample_property: name of the property to format as a DataArray
        :param layer_definition: definition of the way to format the property
        :param nb_keep: the number of different bathymetry estimations to keep for one location.
        :raises IndexError: when the property is not a scalar or a vector
        :returns: an xarray DataArray containing the formatted property
        :raises WavesEstimationAttributeError: when the requested property cannot be found in the
                                               estimations attributes.
        """
        nb_samples = len(self.estimated_bathy)

        dims = layer_definition['dimensions']
        layer_shape: Union[Tuple[int], Tuple[int, int]]
        if 'kKeep' in dims:
            layer_shape = (nb_keep, nb_samples)
        else:
            layer_shape = (nb_samples)
        layer_data = np.full(layer_shape,
                             layer_definition['fill_value'],
                             dtype=layer_definition['data_type'])

        not_found = 0
        for index in range(nb_samples):
            try:
                self._fill_array(sample_property, layer_data, [index])
            except WavesEstimationAttributeError:
                not_found += 1
                continue
        if not_found == nb_samples:
            raise WavesEstimationAttributeError(f'no values defined for: {sample_property}')

        rounded_layer = layer_data.round(decimals=layer_definition['precision'])

        # Add a dimension at the end for time singleton
        array = np.expand_dims(rounded_layer, axis=0)

        return DataArray(array, coords=self._get_coords(dims, nb_keep),
                         dims=dims, attrs=layer_definition['attrs'])

    # TODO: split array filling in two methods: one for 1D (Index) and one for 2D (Index, kKeep)

    def _fill_array(self, sample_property: str, layer_data: np.ndarray, index: List[int]) -> None:
        index = index[0]
        bathymetry_estimations = self.estimated_bathy[index]
        if sample_property == 'x':
            layer_data[index] = np.array([bathymetry_estimations.location.x])
        elif sample_property == 'y':
            layer_data[index] = np.array([bathymetry_estimations.location.y])
        else:
            bathy_property = bathymetry_estimations.get_attribute(sample_property)
            if layer_data.ndim == 1:
                layer_data[index] = np.array(bathy_property)
            else:
                nb_keep = layer_data.shape[0]
                if len(bathy_property) > nb_keep:
                    bathy_property = bathy_property[:nb_keep]
                elif len(bathy_property) < nb_keep:
                    bathy_property += [np.nan] * (nb_keep - len(bathy_property))
                layer_data[:, index] = np.array(bathy_property)

    def _get_coords(self, dims: List[str], nb_keep: int) -> Mapping[Hashable, Any]:
        dict_coords: Dict[Hashable, Any] = {}
        value: Union[np.ndarray, List[datetime]]
        for element in dims:
            if element == 'index':
                value = np.arange(1, len(self.estimated_bathy) + 1)
            elif element == 'kKeep':
                value = np.arange(1, nb_keep + 1)
            elif element == 'time':
                value = self.timestamps
            else:
                raise ValueError('Unknown dimension in netcdf bathy description')
            dict_coords[element] = value
        return dict_coords
