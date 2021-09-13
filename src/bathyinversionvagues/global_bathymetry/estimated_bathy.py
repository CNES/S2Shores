# -*- coding: utf-8 -*-
""" Definition of the EstimatedBathy class

:author: GIROS Alain
:created: 14/05/2021
"""
from datetime import datetime

from typing import Mapping, Hashable, Any, Dict, List

import numpy as np  # @NoMove
from xarray import Dataset, DataArray  # @NoMove
import xarray as xr  # @NoMove

from ..local_bathymetry.waves_fields_estimations import WavesFieldsEstimations


ALL_LAYERS_TYPES = ['NOMINAL', 'DEBUG']

DIMS_Y_X_NKEEP_TIME = ['y', 'x', 'kKeep', 'time']
DIMS_Y_X_TIME = ['y', 'x', 'time']


# TODO: introduce dtype (int for status) and initial value (np.nan or other)
# Provides a mapping from entries into the output dictionary of a local estimator to a netCDF layer.
BATHY_PRODUCT_DEF: Dict[str, Dict[str, Any]] = {
    'sample_status': {'layer_type': ALL_LAYERS_TYPES,
                      'layer_name': 'Status',
                      'dimensions': DIMS_Y_X_TIME,
                      'precision': 8,
                      'attrs': {'Dimension': 'Flags',
                                'name': 'Bathymetry estimation status'}},
    'depth': {'layer_type': ALL_LAYERS_TYPES,
              'layer_name': 'Depth',
              'dimensions': DIMS_Y_X_NKEEP_TIME,
              'precision': 8,
              'attrs': {'Dimension': 'Meters [m]',
                        'name': 'Raw estimated depth'}},
    'direction': {'layer_type': ALL_LAYERS_TYPES,
                  'layer_name': 'Direction',
                  'dimensions': DIMS_Y_X_NKEEP_TIME,
                  'precision': 8,
                  'attrs': {'Dimension': 'degree',
                            'name': 'Wave_direction'}},
    'period': {'layer_type': ALL_LAYERS_TYPES,
               'layer_name': 'Period',
               'dimensions': DIMS_Y_X_NKEEP_TIME,
               'precision': 2,
               'attrs': {'Dimension': 'Seconds [sec]',
                         'name': 'Wave_period'}},
    'celerity': {'layer_type': ALL_LAYERS_TYPES,
                 'layer_name': 'Celerity',
                 'dimensions': DIMS_Y_X_NKEEP_TIME,
                 'precision': 8,
                 'attrs': {'Dimension': 'Meters per second [m/sec]',
                           'name': 'Wave_celerity'}},
    'wavelength': {'layer_type': ALL_LAYERS_TYPES,
                   'layer_name': 'Wavelength',
                   'dimensions': DIMS_Y_X_NKEEP_TIME,
                   'precision': 8,
                   'attrs': {'Dimension': 'Meters [m]',
                             'name': 'Wavelength'}},
    'wavenumber': {'layer_type': ['DEBUG'],
                   'layer_name': 'Wavenumber',
                   'dimensions': DIMS_Y_X_NKEEP_TIME,
                   'precision': 8,
                   'attrs': {'Dimension': 'Per Meter [m-1]',
                             'name': 'Wavenumber'}},
    'distance_to_shore': {'layer_type': ALL_LAYERS_TYPES,
                          'layer_name': 'Distoshore',
                          'dimensions': DIMS_Y_X_TIME,
                          'precision': 8,
                          'attrs': {'Dimension': 'Kilometers [km]',
                                    'name': 'Distance_to_shore'}},
    'delta_celerity': {'layer_type': ALL_LAYERS_TYPES,
                       'layer_name': 'Delta Celerity',
                       'dimensions': DIMS_Y_X_NKEEP_TIME,
                       'precision': 8,
                       'attrs': {'Dimension': 'Meters per seconds2 [m/sec2]',
                                 'name': 'delta_celerity'}},
    'delta_phase': {'layer_type': ['DEBUG'],
                    'layer_name': 'PhaseShift',
                    'dimensions': DIMS_Y_X_NKEEP_TIME,
                    'precision': 8,
                    'attrs': {'Dimension': 'Radians [rd]',
                              'name': 'Phase shift'}},
    'energy_max': {'layer_type': ['DEBUG'],
                   'layer_name': 'Energy',
                   'dimensions': DIMS_Y_X_NKEEP_TIME,
                   'precision': 8,
                   'attrs': {'Dimension': 'Joules per Meter2 [J/m2]',
                             'name': 'Energy'}},
    'gravity': {'layer_type': ['DEBUG'],
                'layer_name': 'Gravity',
                'dimensions': DIMS_Y_X_TIME,
                'precision': 8,
                'attrs': {'Dimension': 'Acceleration [m/s2]',
                          'name': 'Gravity'}},
    '_delta_time': {'layer_type': ['DEBUG'],
                    'layer_name': 'Delta Acquisition Time',
                    'dimensions': DIMS_Y_X_NKEEP_TIME,  # FIXME: does not work with DIMS_Y_X_TIME
                    'precision': 8,
                    'attrs': {'Dimension': 'Duration (s)',
                              'name': 'DeltaTime'}},
    'linearity': {'layer_type': ['DEBUG'],
                  'layer_name': 'Waves Linearity',
                  'dimensions': DIMS_Y_X_NKEEP_TIME,
                  'precision': 8,
                  'attrs': {'Dimension': 'Unitless',
                            'name': 'linearity'}},
    'period_offshore': {'layer_type': ['DEBUG'],
                        'layer_name': 'Period Offshore',
                        'dimensions': DIMS_Y_X_NKEEP_TIME,
                        'precision': 8,
                        'attrs': {'Dimension': 'Seconds [sec]',
                                  'name': 'period_offshore'}},
    'delta_phase_ratio': {'layer_type': ['DEBUG'],
                          'layer_name': 'Delta Phase Ratio',
                          'dimensions': DIMS_Y_X_NKEEP_TIME,
                          'precision': 8,
                          'attrs': {'Dimension': 'Unitless',
                                    'name': 'delta_phase_ratio'}},
    'energy_ratio': {'layer_type': ['DEBUG'],
                     'layer_name': 'Energy Ratio',
                     'dimensions': DIMS_Y_X_NKEEP_TIME,
                     'precision': 8,
                     'attrs': {'Dimension': 'Joules per Meter2 [J/m2]',
                               'name': 'energy_ratio'}},
}


class EstimatedBathy:
    """ This class gathers all the estimated bathymetry samples in a whole dataset.
    """

    def __init__(self, x_samples: np.ndarray, y_samples: np.ndarray,
                 acq_time: str) -> None:
        """ Define dimensions for which the estimated bathymetry samples will be defined.

        :param x_samples: the X coordinates defining the estimated bathymetry samples
        :param y_samples: the Y coordinates defining the estimated bathymetry samples
        :param acq_time: the time at which the bathymetry samples are estimated
        """
        # data is stored as a 2D array of python objects, here a dictionary containing bathy fields.
        self.estimated_bathy = np.empty((y_samples.shape[0], x_samples.shape[0]), dtype=np.object_)

        timestamp = datetime(int(acq_time[:4]), int(acq_time[4:6]), int(acq_time[6:8]),
                             int(acq_time[9:11]), int(acq_time[11:13]), int(acq_time[13:15]))
        self.timestamps = [timestamp]
        self.x_samples = x_samples
        self.y_samples = y_samples

    def store_estimations(self, x_index: int, y_index: int,
                          bathy_estimations: WavesFieldsEstimations) -> None:
        """ Store a set of bathymetry estimations at some location

        :param x_index: index of the sample along the X axis
        :param y_index: index of the sample along the Y axis
        :param bathy_estimations: the whole set of bathy estimations data at one point.
        """
        # TODO: use the x and y coordinates instead of an index, for better modularity
        self.estimated_bathy[y_index, x_index] = bathy_estimations

    def build_dataset(self, layers_type: str, nb_keep: int) -> Dataset:
        """ Build an xarray DataSet containing the estimated bathymetry.

        :param layers_type: select the layers which will be produced in the dataset.
                            Value must be one of ALL_LAYERS_TYPES.
        :param nb_keep: the number of different bathymetry estimations to keep for one location.
        :raises ValueError: when layers_type is not equal to one of the accepted values
        :returns: an xarray Dataset containing the estimated bathymetry.
        """
        if layers_type not in ALL_LAYERS_TYPES:
            msg = f'incorrect layers_type ({layers_type}). Must be one of: {ALL_LAYERS_TYPES}'
            raise ValueError(msg)

        datasets = []
        # make individual dataset with attributes:
        for sample_property, layer_definition in BATHY_PRODUCT_DEF.items():
            if layers_type in layer_definition['layer_type']:
                data_array = self._build_data_array(sample_property, layer_definition, nb_keep)
                datasets.append(data_array.to_dataset(name=layer_definition['layer_name']))

        # Combine all datasets:
        return xr.merge(datasets)

    def _build_data_array(self, sample_property: str,
                          layer_definition: Dict[str, Any], nb_keep: int) -> DataArray:
        """ Build an xarray DataArray containing one estimated bathymetry property.

        :param sample_property: name of the property to format as a DataArray
        :param layer_definition: definition of the way to format the property
        :param nb_keep: the number of different bathymetry estimations to keep for one location.
        :raises IndexError: when the property is not a scalar or a vector
        :returns: an xarray DataArray containing the formatted property
        """
        nb_samples_y, nb_samples_x = self.estimated_bathy.shape

        dims = layer_definition['dimensions']
        if 'kKeep' in dims:
            layer_shape = (nb_samples_y, nb_samples_x, nb_keep)
        else:
            layer_shape = (nb_samples_y, nb_samples_x)
        layer_data = np.full(layer_shape, np.nan)

        for y_index in range(nb_samples_y):
            for x_index in range(nb_samples_x):
                self._fill_array(sample_property, layer_data, y_index, x_index)

        rounded_layer = np.round(layer_data, layer_definition['precision'])
        # Add a dimension at the end for time singleton
        array = np.expand_dims(rounded_layer, axis=rounded_layer.ndim)
        return DataArray(array, coords=self._get_coords(dims, nb_keep),
                         dims=dims, attrs=layer_definition['attrs'])

    # TODO: split array filling in two methods: one for 2D (X, Y) and one for 3D (X, Y, kKeep)
    def _fill_array(self, sample_property: str, layer_data: np.ndarray,
                    y_index: int, x_index: int) -> None:
        waves_fields_estimations = self.estimated_bathy[y_index, x_index]
        if layer_data.ndim == 2:
            nb_keep = 0
        else:
            nb_keep = layer_data.shape[2]

        # TODO: move this logics inside WavesFieldsEstimations class :
        # get_property(sample_property, nb_keep)
        if hasattr(waves_fields_estimations, sample_property):
            # retrieve property from the estimations header
            bathy_property = np.array(getattr(waves_fields_estimations, sample_property))
        else:
            # retrieve property in the list of estimations
            bathy_property = np.full(nb_keep, np.nan)
            try:
                for index, waves_field_estimations in enumerate(waves_fields_estimations):
                    bathy_property[index] = getattr(waves_field_estimations,
                                                    sample_property)
            # FIXME: Should we raise an exception? Parameterize MANDATORY/OPTIONAL ?
            except AttributeError:
                bathy_property = np.array([np.nan])

        if nb_keep == 0:
            layer_data[y_index, x_index] = bathy_property
        else:
            layer_data[y_index, x_index, :] = bathy_property

    def _get_coords(self, dims: List[str], nb_keep: int) -> Mapping[Hashable, Any]:
        dict_coords: Mapping[Hashable, Any] = {}
        for element in dims:
            if element == 'y':
                value = self.y_samples
            elif element == 'x':
                value = self.x_samples
            elif element == 'kKeep':
                value = np.arange(1, nb_keep + 1)
            else:
                value = self.timestamps
            dict_coords[element] = value
        return dict_coords
