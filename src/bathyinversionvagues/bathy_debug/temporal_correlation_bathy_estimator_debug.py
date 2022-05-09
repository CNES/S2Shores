# -*- coding: utf-8 -*-
""" Class performing bathymetry computation using temporal correlation method

:author: Degoul Romain
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 18/06/2021
"""

import os
import numpy as np
from copy import deepcopy
from typing import Optional, TYPE_CHECKING

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from ..local_bathymetry.temporal_correlation_bathy_estimator import TemporalCorrelationBathyEstimator
from ..image.image_geometry_types import PointType
from ..image.ortho_sequence import OrthoSequence
from ..waves_exceptions import WavesEstimationError, NotExploitableSinogram, CorrelationComputationError

from .local_bathy_estimator_debug import LocalBathyEstimatorDebug

if TYPE_CHECKING:
    from ..global_bathymetry.bathy_estimator import BathyEstimator  # @UnusedImport


class TemporalCorrelationBathyEstimatorDebug(LocalBathyEstimatorDebug,
                                             TemporalCorrelationBathyEstimator):
    """ Class performing debugging for temporal correlation method
    """

    def __init__(self, location: PointType, ortho_sequence: OrthoSequence,
                 global_estimator: 'BathyEstimator',
                 selected_directions: Optional[np.ndarray] = None) -> None:
        # FIXME: Handle severals wave_estimations
        ######################################################
        super().__init__(location, ortho_sequence, global_estimator, selected_directions)
        self._figure = plt.figure(constrained_layout=True)
        self._nb_estimation = 0
        self._gs = None

    def run(self) -> None:
        try:
            super().run()
        except WavesEstimationError as excp:
            self.explore_results()
            raise excp
        except NotExploitableSinogram as excp:
            self.initialize_figure()
            self.show_thumbnail()
            self.show_correlation_matrix()
            self.show_radon_matrix()
            self.show_failed_sinogram()
            self.dump_figure()
            raise excp
        except CorrelationComputationError as excp:
            self.initialize_figure()
            self.show_thumbnail()
            self.print_correlation_matrix_error()
            self.dump_figure()
            raise excp

    def initialize_figure(self):
        self._nb_estimation = len(self.metrics['direction_estimations'])
        self._gs = gridspec.GridSpec(2 + 2 * self._nb_estimation, 3, figure=self._figure)

    def show_thumbnail(self) -> None:
        """ Show first frame in sequence for a debug point
        """
        # First diagram : first image of the sequence
        first_image = self.ortho_sequence[0].pixels
        subfigure = self._figure.add_subplot(self._gs[0, 0])
        imin = np.min(first_image)
        imax = np.max(first_image)
        subfigure.imshow(first_image, norm=Normalize(vmin=imin, vmax=imax))
        (l_1, l_2) = np.shape(first_image)
        radius = min(l_1, l_2) / 3
        if 'directions' in self.metrics:
            for direction in self.metrics['directions']:
                cartesian_dir_x = np.cos(np.deg2rad(direction))
                cartesian_dir_y = -np.sin(np.deg2rad(direction))
                subfigure.arrow(
                    l_1 // 2,
                    l_2 // 2,
                    radius * cartesian_dir_x,
                    radius * cartesian_dir_y)
        plt.title('Thumbnail')

    def show_correlation_matrix(self) -> None:
        """ Show correlation matrix for a debug point
        """
        # Second diagram : correlation matrix
        subfigure = self._figure.add_subplot(self._gs[0, 1])
        imin = np.min(self.correlation_image.pixels)
        imax = np.max(self.correlation_image.pixels)
        subfigure.imshow(self.correlation_image.pixels, norm=Normalize(vmin=imin, vmax=imax))
        (l_1, l_2) = np.shape(self.correlation_image.pixels)
        radius = min(l_1, l_2) / 3
        if 'directions' in self.metrics:
            for direction in self.metrics['directions']:
                cartesian_dir_x = np.cos(np.deg2rad(direction))
                cartesian_dir_y = -np.sin(np.deg2rad(direction))
                subfigure.arrow(
                    l_1 // 2,
                    l_2 // 2,
                    radius * cartesian_dir_x,
                    radius * cartesian_dir_y)
        plt.title('Correlation matrix')

    def show_radon_matrix(self) -> None:
        """ Show radon matrix for a debug point
        """
        # Third diagram : Radon transform & maximum variance
        subfigure = self._figure.add_subplot(self._gs[1, :2])
        radon_array, _ = self.metrics['radon_transform'].get_as_arrays()
        nb_directions, _ = radon_array.shape
        directions = self.selected_directions
        min_dir = np.min(directions)
        max_dir = np.max(directions)
        subfigure.imshow(radon_array, interpolation='nearest', aspect='auto',
                         origin='lower', extent=[min_dir, max_dir, 0, nb_directions])
        l_1, _ = np.shape(radon_array)
        plt.plot(self.selected_directions, l_1 * self._metrics['variances'] /
                 np.max(self._metrics['variances']), 'r')
        if 'directions' in self.metrics:
            for direction in self.metrics['directions']:
                subfigure.arrow(direction, 0, 0, l_1)
                plt.annotate(f"{direction} °",
                             (direction + 5, 10), color='orange')
        plt.title('Radon matrix')

    def show_sinogram(self) -> None:
        """ Show sinogram for a debug point
        """
        # Fourth diagram : Sinogram & wave length computation
        list_infos_sinogram = self.metrics['list_infos_sinogram']
        nb_sinograms = 0
        for sinogram_max_var, wave_length_zeros, max_indices in list_infos_sinogram:
            subfigure = self._figure.add_subplot(self._gs[2 + nb_sinograms, :2])
            x_axis = np.arange(-(len(sinogram_max_var) // 2), len(sinogram_max_var) // 2 + 1)
            if sinogram_max_var is not None:
                subfigure.plot(x_axis, sinogram_max_var)
            if wave_length_zeros is not None:
                subfigure.plot(x_axis[wave_length_zeros],
                               sinogram_max_var[wave_length_zeros], 'ro')
            if max_indices is not None:
                subfigure.plot(x_axis[max_indices],
                               sinogram_max_var[max_indices], 'go')
            nb_sinograms = nb_sinograms + 1
        plt.title('Sinogram')

    def show_failed_sinogram(self) -> None:
        """ Show sinogram on which computation has failed
        """
        # Fourth diagram : Sinogram & wave length computation
        subfigure = self._figure.add_subplot(self._gs[2, :2])
        sinogram_max_var = self.metrics['sinogram_max_var']
        x_axis = np.arange(-(len(sinogram_max_var) // 2), len(sinogram_max_var) // 2 + 1)
        subfigure.plot(x_axis, sinogram_max_var)

    def show_values(self) -> None:
        """ Show physical values for a debug point
        """
        # Fifth  diagram
        nb_sinograms = 0
        direction_estimations = self.metrics['direction_estimations']
        for direction_estimation in direction_estimations:
            if direction_estimation:
                subfigure = self._figure.add_subplot(self._gs[2 + nb_sinograms, 2])
                subfigure.axis('off')
                celerities = direction_estimation.get_attribute('celerity')
                celerities = [round(celerity, 2) for celerity in celerities]
                distances = direction_estimation.get_attribute('delta_position')
                linerities = direction_estimation.get_attribute('linearity')
                linerities = [round(linearity, 2) for linearity in linerities]
                direction_estimation_tmp = deepcopy(direction_estimation)
                direction_estimation.remove_unphysical_wave_fields()
                direction_estimation.sort_on_attribute('linearity', reverse=False)
                if direction_estimation:
                    estimation = direction_estimation[0]
                    subfigure.annotate(f'direction = {estimation.direction}\n'
                                       f'wave_length = {estimation.wavelength}\n'
                                       f' dx = {distances} \n'
                                       f' c = {celerities} \n ckg = {linerities}\n'
                                       f' chosen_celerity = {estimation.celerity}\n'
                                       f' depth = {estimation.depth}',
                                       (0, 0), color='g')
                else:
                    estimation = direction_estimation_tmp[0]
                    subfigure.annotate(f'direction = {estimation.direction}\n'
                                       f'wave_length = {estimation.wavelength} \n'
                                       f' dx = {distances} \n'
                                       f' c = {celerities} \n ckg = {linerities}\n'
                                       f' No estimations have been found', (0, 0), color='g')
                nb_sinograms = nb_sinograms + 1

    def print_correlation_matrix_error(self) -> None:
        """ Display a message for correlation matrix error in debug image
        """
        subfigure = self._figure.add_subplot(self._gs[1, :2])
        subfigure.axis('off')
        subfigure.annotate('Correlation can not be computed',
                           (0, 0), color='g')

    def dump_figure(self) -> None:
        """ Save figure for a debug point
        """
        if self.global_estimator.debug_path:
            self._figure.savefig(
                os.path.join(
                    self.global_estimator.debug_path,
                    f'Infos_point_{self.location[0]}_{self.location[1]}.png'),
                dpi=300)
        plt.close()

    def explore_results(self) -> None:
        """ Full routine for debugging point
        """
        self.initialize_figure()
        self.show_thumbnail()
        self.show_correlation_matrix()
        self.show_radon_matrix()
        self.show_sinogram()
        self.show_values()
        self.dump_figure()
