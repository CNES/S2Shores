# -*- coding: utf-8 -*-
""" Definition of the BathyEstimator abstract class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 17/05/2021
"""
from pathlib import Path
from typing import Any, Dict, List, Optional  # @NoMove

import numpy as np
import xarray as xr  # @NoMove
from numpy import dtype
from shapely.geometry import Point
from xarray import Dataset  # @NoMove

from ..image.image_geometry_types import MarginsType
from ..image.ortho_stack import FramesIdsType, OrthoStack
from ..image.sampled_ortho_image import SampledOrthoImage
from .bathy_estimator_parameters import BathyEstimatorParameters
from .bathy_estimator_providers import BathyEstimatorProviders
from .ortho_bathy_estimator import OrthoBathyEstimator


class BathyEstimator(BathyEstimatorParameters, BathyEstimatorProviders):
    """ Management of bathymetry computation and parameters on a single product. Computation
    is split in several cartographic tiles, which must be run separately, either in parallel or
    sequentially.
    """

    def __init__(self, ortho_stack: OrthoStack, wave_params: Dict[str, Any], output_dir: Path,
                 nb_subtiles_max: int = 1) -> None:
        """Create a BathyEstimator object and set necessary informations

        :param ortho_stack: the orthorectified stack onto which bathymetry must be estimated.
        :param wave_params: parameters for the global and local bathymetry estimators
        :param output_dir: path to the directory where the netCDF bathy file will be written.
        :param nb_subtiles_max: Nb of subtiles for bathymetry estimation
        """
        BathyEstimatorParameters.__init__(self, wave_params)
        BathyEstimatorProviders.__init__(self, ortho_stack)
        # Store arguments in attributes for further use
        self._output_dir = output_dir

        # Create subtiles onto which bathymetry estimation will be done
        self._nb_subtiles_max = nb_subtiles_max
        self.subtiles: List[SampledOrthoImage]

        # Init debugging points handling
        self._debug_path: Optional[Path] = None
        self._debug_samples: List[Point] = []
        self._debug_sample = False

    @property
    def smoothing_requested(self) -> bool:
        """ :returns: True if both smoothing columns and lines parameters are non zero
        """
        return self.smoothing_columns_size != 0 and self.smoothing_lines_size != 0

    @property
    def measure_extent(self) -> MarginsType:
        """ :returns: the cartographic extent to be used for bathy estimation around a point
        """
        return (self.window_size_x / 2., self.window_size_x / 2.,
                self.window_size_y / 2., self.window_size_y / 2.)

    @property
    def selected_frames(self) -> FramesIdsType:
        """ :returns: the list of frames selected for running the estimation, or the list of all
                      the usable frames if not specified in the parameters.
        """
        selected_frames = self.selected_frames_param
        if selected_frames is None:
            selected_frames = self._ortho_stack.usable_frames
        return selected_frames

    @property
    def nb_subtiles(self) -> int:
        """ :returns: the number of subtiles
        """
        return len(self.subtiles)

    def create_subtiles(self) -> None:
        """ Warmup of the bathy estimator by creating the processing subtiles
        """
        roi = None
        if self._roi_provider is not None and self._limit_to_roi:
            roi = self._roi_provider.bounding_box(0.1)
        self.subtiles = self._ortho_stack.build_subtiles(self._nb_subtiles_max,
                                                         self.sampling_step_x,
                                                         self.sampling_step_y,
                                                         self.measure_extent,
                                                         roi=roi)

    def compute_bathy_for_subtile(self, subtile_number: int) -> Dataset:
        """ Computes the bathymetry dataset for a given subtile.

        :param subtile_number: number of the subtile
        :returns: Subtile dataset
        """
        # Retrieve the subtile.
        subtile = self.subtiles[subtile_number]
        print(f'Subtile {subtile_number}: {self._ortho_stack.short_name} {subtile}')

        # Build a bathymertry estimator over the subtile and launch estimation.
        subtile_estimator = OrthoBathyEstimator(self, subtile)
        dataset = subtile_estimator.compute_bathy()

        # Build the bathymetry dataset for the subtile.
        # Add spatial_ref variable to the Dataset
        dataset = dataset.assign({'spatial_ref': 0})
        # Assign relevant projection attribute of the spatial_ref variable
        dataset.spatial_ref.attrs['spatial_ref'] = self._ortho_stack.build_spatial_ref()

        # necessary to have a correct georeferencing
        if 'x' in dataset.coords : # only if output_format is GRID
            dataset.x.attrs['standard_name'] = "projection_x_coordinate"
            dataset.y.attrs['standard_name'] = "projection_y_coordinate"

        infos = self.build_infos()
        infos.update(self._ortho_stack.build_infos())
        for key, value in infos.items():
            dataset.attrs[key] = value

        # We return the dataset instead of storing it in the instance, for multiprocessing reasons.
        return dataset

    def merge_subtiles(self, bathy_subtiles: List[Dataset]) -> None:
        """Merge all the subtiles datasets in memory into a single one in a netCDF file

        :param bathy_subtiles: Subtiles datasets
        """
        merged_bathy = xr.combine_by_coords(bathy_subtiles)
        product_name = self._ortho_stack.full_name
        netcdf_output_path = (self._output_dir / product_name).with_suffix('.nc')
        try:
            merged_bathy.to_netcdf(path=netcdf_output_path, format='NETCDF4')
        except Exception as e:
            print(f'ERROR while merging subtiles of {product_name}, product not saved. {e}')

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this estimator
        """

        title = 'Wave parameters and raw bathymetry derived from satellite imagery.'
        title += ' No tidal vertical adjustment.'
        infos = {'title': title,
                 'institution': 'CNES-LEGOS',
                 'coordinates': 'spatial_ref'}

        # metadata from the parameters
        infos['waveEstimationMethod'] = self.local_estimator_code
        infos['ChainVersions'] = self.chains_versions
        infos['Resolution X'] = str(self.sampling_step_x)
        infos['Resolution Y'] = str(self.sampling_step_y)
        infos['OffshoreLimit_kms'] = str(self.max_offshore_distance)
        return infos

# ++++++++++++++++++++++++++++ Debug support +++++++++++++++++++++++++++++
    @property
    def debug_path(self) -> Optional[Path]:
        """ :returns: path to a directory where debugging info can be written.
        """
        return self._debug_path

    @debug_path.setter
    def debug_path(self, path: Path) -> None:
        self._debug_path = path

    def set_debug_area(self, bottom_left_corner: Point, top_right_corner: Point,
                       decimation: int) -> None:
        """ Sets all points within rectangle defined by bottom_left_corner and top_right_corner to
        debug

        :param bottom_left_corner: point defining the bottom left corner of the area of interest
        :param top_right_corner: point defining the top right corner of the area of interest
        :param decimation: decimation factor for all points within the area of interest
                           (oversize factor will lead to a single point)

        """
        x_samples = np.array([])
        y_samples = np.array([])
        for subtile in self.subtiles:
            x_samples = np.concatenate((x_samples, subtile.carto_sampling.x_samples))
            y_samples = np.concatenate((y_samples, subtile.carto_sampling.y_samples))
        x_samples_filtered = x_samples[np.logical_and(
            x_samples > bottom_left_corner.x, x_samples < top_right_corner.x)][::decimation]
        y_samples_filtered = y_samples[np.logical_and(
            y_samples > bottom_left_corner.y, y_samples < top_right_corner.y)][::decimation]
        list_samples = []
        for x_coord in x_samples_filtered:
            for y_coord in y_samples_filtered:
                list_samples.append(Point(x_coord, y_coord))
        self._debug_samples = list_samples

    def set_debug_samples(self, samples: List[Point]) -> None:
        """ Sets the list of sample points to debug

        :param samples: a list of (X,Y) tuples defining the points to debug
        """
        self._debug_samples = []
        for sample in samples:
            if self._ortho_stack.is_window_inside(sample, self.measure_extent):
                self._debug_samples.append(sample)
            else:
                print(f'{sample} is not in roi-window_size/2.')
        if self._debug_samples==[]:
            raise ValueError("There is no point available to debug. Check your points' coordinates and the window size.")

    def set_debug_flag(self, sample: Point) -> None:
        """ Set or reset the debug flag for a given point depending on its presence into the set
        of points to debug.

        :param sample: The point for which the debug flag must be set
        """
        self._debug_sample = sample in self._debug_samples

    @property
    def debug_sample(self) -> bool:
        """ :returns: the current value of the debugging flag
        """
        return self._debug_sample
