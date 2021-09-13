# -*- coding: utf-8 -*-
""" Definition of the OrthoBathyEstimator class

:author: GIROS Alain
:created: 05/05/2021
"""
import time
from typing import Dict, List, TYPE_CHECKING
import warnings

from ..image.sampled_ortho_image import SampledOrthoImage
from ..image_processing.waves_image import WavesImage
from ..local_bathymetry.local_bathy_estimator import LocalBathyEstimator
from ..local_bathymetry.local_bathy_estimator_factory import local_bathy_estimator_factory
from ..local_bathymetry.waves_fields_estimations import WavesFieldsEstimations
from ..waves_exceptions import WavesException
from .estimated_bathy import EstimatedBathy


import numpy as np  # @NoMove
from xarray import Dataset  # @NoMove


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

        :return: Estimated bathymetry dataset
        """

        start_load = time.time()
        # nbkeep shall be understood as a filtering in terms of the number of proposed samples.
        # Will disappear when true Waves Fields will be identified and implemented.
        nb_keep = self.parent_estimator.waveparams.NKEEP

        estimated_bathy = EstimatedBathy(self.sampled_ortho.x_samples, self.sampled_ortho.y_samples,
                                         self.sampled_ortho.image.acquisition_time)

        # subtile reading
        sub_tile_images: List[np.ndarray] = []
        for band_id in self.parent_estimator.bands_identifiers:
            sub_tile_images.append(self.sampled_ortho.read_pixels(band_id))
        print(f'Loading time: {time.time() - start_load:.2f} s')

        start = time.time()
        in_water_points = 0
        for i, x_sample in enumerate(self.sampled_ortho.x_samples):
            for j, y_sample in enumerate(self.sampled_ortho.y_samples):
                self.parent_estimator.set_debug((x_sample, y_sample))

                distance = self.parent_estimator.get_distoshore((x_sample, y_sample))
                gravity = self.parent_estimator.get_gravity((x_sample, y_sample), 0.)

                bathy_estimations = WavesFieldsEstimations()
                bathy_estimations.distance_to_shore = distance
                bathy_estimations.gravity = gravity
                # do not compute on land
                # FIXME: distance to shore test should take into account windows sizes
                if distance > 0:
                    in_water_points += 1
                    # computes the bathymetry at the specified position
                    local_bathy_estimator = self._compute_local_bathy(sub_tile_images,
                                                                      x_sample, y_sample)
                    local_bathy_estimator.validate_waves_fields()
                    local_bathy_estimator.sort_waves_fields()
                    waves_fields_estimations = local_bathy_estimator.waves_fields_estimations
                else:
                    waves_fields_estimations = []
                print(len(waves_fields_estimations))

                # TODO: do this filtering in build_dataset()
                # Keep only a limited number of waves fields and bathy estimations
                while len(waves_fields_estimations) > nb_keep:
                    waves_fields_estimations.pop()

                # Store bathymetry sample
                # TODO: retrieve right object directly from estimator
                for wave_field in waves_fields_estimations:
                    bathy_estimations.append(wave_field)
                self.parent_estimator.print_estimations_debug(bathy_estimations,
                                                              'after estimations sorting')

                estimated_bathy.store_estimations(i, j, bathy_estimations)

        total_points = self.sampled_ortho.nb_samples
        comput_time = time.time() - start
        print(f'Computed {in_water_points}/{total_points} points in: {comput_time:.2f} s')

        return estimated_bathy.build_dataset(self.parent_estimator.waveparams.LAYERS_TYPE, nb_keep)

    def _compute_local_bathy(self, sub_tile_images: List[np.ndarray],
                             x_sample: float, y_sample: float) -> LocalBathyEstimator:

        window = self.sampled_ortho.window_extent((x_sample, y_sample))
        # TODO: Link WavesImage to OrthoImage and use resolution from it?
        resolution = self.sampled_ortho.image.spatial_resolution

        # Create the sequence of WavesImages (to be used by ALL estimators)
        images_sequence: List[WavesImage] = []
        for index, band_id in enumerate(self.parent_estimator.bands_identifiers):
            window_image = WavesImage(sub_tile_images[index][window[0]:window[1] + 1,
                                                             window[2]:window[3] + 1],
                                      resolution)
            images_sequence.append(window_image)
            if self.parent_estimator.debug_sample:
                print(f'Subtile shape {sub_tile_images[index].shape}')
                print(f'Window in ortho image coordinate: {window}')
                print(f'--{band_id} imagette {window_image.pixels.shape}:')
                print(window_image.pixels)

        # Local bathymetry computation
        # TODO: define the class to use only once for global estimator (no change between samples)
        # FIXME: global inforamtion on this point should be passed to the factory (gravity, ...)
        local_bathy_estimator = local_bathy_estimator_factory(images_sequence,
                                                              self.parent_estimator)
        # FIXME: this is not clean
        # local_bathy_estimator.waves_fields_estimations = self.bathy_estimations
        local_bathy_estimator.set_position((x_sample, y_sample))

        try:
            local_bathy_estimator.run()
        except WavesException as excp:
            warnings.warn(f'Unable to estimate bathymetry: {str(excp)}')

        return local_bathy_estimator

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this estimator
        """

        title = 'Wave parameters and raw bathymetry derived from satellite imagery.'
        title += ' No tidal vertical adjustment.'
        infos = {'title': title,
                 'institution': 'CNES-LEGOS'}

        # metadata from the parameters
        infos['waveEstimationMethod'] = self.parent_estimator.waveparams.WAVE_EST_METHOD
        infos['ChainVersions'] = self.parent_estimator.waveparams.CHAINS_VERSIONS

        return infos
