# -*- coding: utf-8 -*-
""" Definition of the OrthoBathyEstimator class

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 05/05/2021
"""
import time
import warnings
from typing import TYPE_CHECKING  # @NoMove

from shapely.geometry import Point
from xarray import Dataset  # @NoMove

from ..data_model.bathymetry_sample_estimations import BathymetrySampleEstimations
from ..data_model.estimated_carto_bathy import EstimatedCartoBathy
from ..data_model.estimated_points_bathy import EstimatedPointsBathy
from ..image.ortho_sequence import OrthoSequence
from ..image.sampled_ortho_image import SampledOrthoImage
from ..local_bathymetry.local_bathy_estimator_factory import local_bathy_estimator_factory
from ..waves_exceptions import WavesException

if TYPE_CHECKING:
    from .bathy_estimator import BathyEstimator  # @UnusedImport


# TODO: Make this class inherit from BathyEstimator ?
class OrthoBathyEstimator:
    """ This class implements the computation of bathymetry over a sampled orthorectifed image.
    """

    def __init__(self, estimator: 'BathyEstimator', sampled_ortho: SampledOrthoImage) -> None:
        """ Constructor

        :param estimator: the parent estimator of this estimator
        :param sampled_ortho: the image onto which the bathy estimation must be done
        """
        self.sampled_ortho = sampled_ortho
        self.parent_estimator = estimator

    def compute_bathy(self) -> Dataset:
        """ Computes the bathymetry dataset for the samples belonging to a given subtile.

        :raises ValueError: when output format is not a supported one
        :return: Estimated bathymetry dataset
        """

        start_load = time.time()
        # nbkeep shall be understood as a filtering in terms of the number of proposed samples.
        # Will disappear when true Wave Fields will be identified and implemented.
        nb_keep = self.parent_estimator.nb_max_wave_fields

        # subtile reading
        sub_tile_images = OrthoSequence(self.parent_estimator.delta_time_provider)
        for frame_id in self.parent_estimator.selected_frames:
            sub_tile_images.append_image(self.sampled_ortho.read_frame_image(frame_id), frame_id)
        print(f'Loading time: {time.time() - start_load:.2f} s')

        start = time.time()
        computed_points = 0

        if self.parent_estimator.output_format == 'POINT':
            # Estimate bathy on points
            total_points = len(self.parent_estimator._debug_samples)
            estimated_bathy = EstimatedPointsBathy(total_points,
                                                   self.sampled_ortho.ortho_stack.acquisition_time)
            samples = self.parent_estimator._debug_samples

        elif self.parent_estimator.output_format == 'GRID':
            # Estimate bathy over a grid
            estimated_bathy = EstimatedCartoBathy(self.sampled_ortho.carto_sampling,
                                                  self.sampled_ortho.ortho_stack.acquisition_time)
            samples = self.sampled_ortho.carto_sampling.x_y_sampling()
            total_points = self.sampled_ortho.carto_sampling.nb_samples
        else:
            raise ValueError('Output format must be one of the proposed ones in the config file.')

        for index, sample in enumerate(samples):
            bathy_estimations = self._run_local_bathy_estimator(sub_tile_images, sample)
            if bathy_estimations.distance_to_shore > 0 and bathy_estimations.inside_roi:
                computed_points += 1
            # Store bathymetry sample estimations
            estimated_bathy.store_estimations(index, bathy_estimations)

        comput_time = time.time() - start
        print(f'Computed {computed_points}/{total_points} points in: {comput_time:.2f} s')

        return estimated_bathy.build_dataset(self.parent_estimator.layers_type, nb_keep)

    def _run_local_bathy_estimator(self, sub_tile_images: OrthoSequence,
                                   estimation_point: Point) -> BathymetrySampleEstimations:

        self.parent_estimator.set_debug_flag(estimation_point)

        # computes the bathymetry at the specified position
        try:
            # Build the images sequence for the estimation point
            window = self.sampled_ortho.window_extent(estimation_point)
            ortho_sequence = sub_tile_images.extract_window(window)
            if self.parent_estimator.debug_sample:
                for index, image_sequence in enumerate(ortho_sequence):
                    print(f'Subtile shape {sub_tile_images[index].pixels.shape}')
                    print(f'Window inside ortho image coordinates: {window}')
                    print(f'--{ortho_sequence._images_id[index]} imagette {image_sequence}')

            # TODO: use selected_directions argument
            local_bathy_estimator = local_bathy_estimator_factory(estimation_point, ortho_sequence,
                                                                  self.parent_estimator)

            bathy_estimations = local_bathy_estimator.bathymetry_estimations
            if local_bathy_estimator.can_estimate_bathy():
                local_bathy_estimator.run()
                if self.parent_estimator._debug_sample:
                    bathy_estimations.flag_unphysical_wave_fields()
                else:
                    bathy_estimations.remove_unphysical_wave_fields()
                bathy_estimations.sort_on_attribute(local_bathy_estimator.final_estimations_sorting)
                if self.parent_estimator.debug_sample:
                    print('estimations after sorting :')
                    print(bathy_estimations)
        except WavesException as excp:
            warn_msg = f'Unable to estimate bathymetry: {str(excp)}'
            warnings.warn(warn_msg)
            bathy_estimations = local_bathy_estimator.bathymetry_estimations
            bathy_estimations.clear()
        return bathy_estimations
