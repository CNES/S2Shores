# -*- coding: utf-8 -*-
import os
from typing import TYPE_CHECKING  # @NoMove

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

from ..waves_exceptions import DebugDisplayError
from ..bathy_physics import depth_from_dispersion


if TYPE_CHECKING:
    from ..local_bathymetry.temporal_correlation_bathy_estimator import \
        TemporalCorrelationBathyEstimator


def temporal_method_debug(temporal_estimator: 'TemporalCorrelationBathyEstimator') -> None:
    # FIXME : Handle severals wave_estimations
    wave_estimation = temporal_estimator.waves_fields_estimations[0]
    wave_direction = wave_estimation.direction
    wave_wavelength = wave_estimation.wavelength
    wave_celerity = wave_estimation.celerity
    wave_period = wave_estimation.period

    # Note that wave direction is clockwise origin east
    px = np.cos(np.deg2rad(wave_direction))
    py = -np.sin(np.deg2rad(wave_direction))
    first_image = temporal_estimator.images_sequence[0].pixels
    correlation_matrix = temporal_estimator.correlation_image.pixels
    sinogram_max_var = temporal_estimator.metrics['sinogram_max_var']
    x = temporal_estimator.metrics['x']
    interval = temporal_estimator._metrics['interval']

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(5, 2, figure=fig)

    # First diagram : first image of the sequence
    ax = fig.add_subplot(gs[0, 0])
    imin = np.min(first_image)
    imax = np.max(first_image)
    ax.imshow(first_image, norm=Normalize(vmin=imin, vmax=imax))
    (l1, l2) = np.shape(first_image)
    radius = min(l1, l2) / 3
    ax.arrow(l1 // 2, l2 // 2, radius * px, radius * py)
    plt.title('Thumbnail')

    # Second diagram : correlation matrix
    ax2 = fig.add_subplot(gs[0, 1])
    imin = np.min(correlation_matrix)
    imax = np.max(correlation_matrix)
    ax2.imshow(correlation_matrix, norm=Normalize(vmin=imin, vmax=imax))
    (l1, l2) = np.shape(correlation_matrix)
    radius = min(l1, l2) / 3
    ax2.arrow(l1 // 2, l2 // 2, radius * px, radius * py)
    plt.title('Correlation matrix')

    # Third diagram : Radon transform & maximum variance
    ax3 = fig.add_subplot(gs[1, :2])
    radon_array, _ = temporal_estimator.metrics['radon_transform'].get_as_arrays()
    ax3.imshow(radon_array, interpolation='nearest', aspect='auto', origin='lower')
    (l1, l2) = np.shape(radon_array)
    plt.plot(temporal_estimator.selected_directions, l1 * temporal_estimator.metrics['variances'] /
             np.max(temporal_estimator.metrics['variances']), 'r')
    ax3.arrow(wave_direction, 0, 0, l1)
    plt.annotate('%d °' % wave_direction, (wave_direction + 5, 10), color='orange')
    plt.title('Radon matrix')

    # Fourth diagram : Sinogram & wave length computation
    ax4 = fig.add_subplot(gs[2, :2])

    length_signal = len(sinogram_max_var)
    ax4.plot(x, sinogram_max_var)
    ax4.scatter(x[interval], sinogram_max_var[interval], s=4 *
                mpl.rcParams['lines.markersize'], c='orange')
    min_limit_x = np.min(x)
    min_limit_y = np.min(sinogram_max_var)
    ax4.plot(x[temporal_estimator._metrics['wave_length_zeros']],
             sinogram_max_var[temporal_estimator._metrics['wave_length_zeros']], 'ro')
    ax4.plot(x[temporal_estimator._metrics['max_indices']],
             sinogram_max_var[temporal_estimator._metrics['max_indices']], 'go')

    bathy = depth_from_dispersion(1 / wave_estimation.wavelength,
                                  wave_estimation.celerity, temporal_estimator.gravity)
    ax4.annotate('depth = {:.2f}'.format(bathy), (min_limit_x, min_limit_y), color='orange')
    plt.title('Sinogram')

    # Fifth  diagram
    ax5 = fig.add_subplot(gs[3, :2])
    ax5.axis('off')
    dephasings = temporal_estimator._metrics['dephasings']
    celerities = temporal_estimator._metrics['celerities']
    celerities_from_periods = temporal_estimator._metrics['celerities_from_periods']
    chain_dx = ' '.join([f'{dephasing:.2f} | ' for dephasing in dephasings])
    chain_celerities = ' '.join([f'{celerity:.2f} | ' for celerity in celerities])
    chain_celerities_from_period = ' '.join(
        [f'{celerity_from_period:.2f} | ' for celerity_from_period in celerities_from_periods])
    ax5.annotate(
        f'wave_length = {wave_wavelength} \n dx = {chain_dx} \n c = {chain_celerities} \n c_from_period = {chain_celerities_from_period}\n chosen_celerity = {wave_celerity}', (0, 0), color='g')

    # sixth  diagram : Temporal reconstruction
    fig_temporal_signals = plt.figure('Signaux temporal', constrained_layout=True)
    hops_number = len(temporal_estimator._metrics['temporal_signals'])
    gs = gridspec.GridSpec(hops_number, 1, figure=fig_temporal_signals)
    for i in range(hops_number):
        temporal_signal = temporal_estimator._metrics['temporal_signals'][i]
        arg_peak_max = temporal_estimator._metrics['arg_peaks_max'][i]
        dephasing = temporal_estimator._metrics['dephasings'][i]
        temporal_period = temporal_estimator._metrics['periods'][i]
        celerities_from_periods = temporal_estimator._metrics['celerities_from_periods'][i]
        ax = fig_temporal_signals.add_subplot(gs[i, :])
        ax.plot(temporal_signal)
        ax.plot(arg_peak_max, temporal_signal[arg_peak_max], 'ro')
        ax.annotate('T={:.2f} s  | c = L/T = {:.2f}/{:.2f} = {:.2f}'.format(temporal_period, wave_wavelength, temporal_period, celerities_from_periods),
                    (0, np.min(temporal_signal)), color='r')

    fig_temporal_signals.savefig(os.path.join(temporal_estimator.local_estimator_params[DEBUG_PATH],
                                              f'Temporal_signals_{temporal_estimator.location[0]}_{temporal_estimator.location[1]}.png'), dpi=300)
    plt.close(fig_temporal_signals)
    fig.savefig(os.path.join(temporal_estimator.local_estimator_params[DEBUG_PATH],
                             f'Infos_point_{temporal_estimator.location[0]}_{temporal_estimator.location[1]}.png'), dpi=300)
