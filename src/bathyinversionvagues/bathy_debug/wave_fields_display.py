# -*- coding: utf-8 -*-
"""
Class managing the computation of wave fields from two images taken at a small time interval.


:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 5 mars 2021
"""
from typing import Optional, List, Tuple, TYPE_CHECKING  # @NoMove


from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from bathyinversionvagues.image_processing.waves_radon import WavesRadon
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


if TYPE_CHECKING:
    from ..local_bathymetry.spatial_dft_bathy_estimator \
        import SpatialDFTBathyEstimator  # @UnusedImport


def display_curve(data: np.ndarray, legend: str) -> None:
    _, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(legend)
    plt.show()


def display_3curves(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray) -> None:
    _, ax = plt.subplots(3)
    ax[0].plot(data1)
    ax[1].plot(data2)
    ax[2].plot(data3)
    plt.show()


def display_4curves(data1: np.ndarray, data2: np.ndarray, data3: np.ndarray,
                    data4: np.ndarray) -> None:
    _, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(data1)
    ax[1, 0].plot(data2)
    ax[0, 1].plot(data3)
    ax[1, 1].plot(data4)
    plt.show()


def display_image(data: np.ndarray, legend: str) -> None:
    _, ax = plt.subplots()
    ax.imshow(data, aspect='auto', cmap='gray')
    ax.set_title(legend)
    plt.show()


def get_display_title(local_estimator: 'SpatialDFTBathyEstimator') -> str:
    title = f'{local_estimator.global_estimator._ortho_stack.short_name} {local_estimator.location}'
    return title


def build_image_display(axes: Axes, title: str, image: np.ndarray,
                        directions: Optional[List[Tuple[float, float]]] = None,
                        cmap: Optional[str] = None) -> None:
    imin = np.min(image)
    imax = np.max(image)
    axes.imshow(image, norm=Normalize(vmin=imin, vmax=imax), cmap=cmap)
    (l1, l2) = np.shape(image)
    radius = min(l1, l2) / 2
    if directions is not None:
        for direction, coeff_length in directions:
            arrow_length = radius * coeff_length
            dir_rad = np.deg2rad(direction)
            axes.arrow(l1 // 2, l2 // 2,
                       np.cos(dir_rad) * arrow_length, -np.sin(dir_rad) * arrow_length,
                       head_width=2, head_length=3, color='r')
    axes.set_title(title)


def build_directional_2d_display(axes: Axes, title: str, values: np.ndarray,
                                 directions: np.ndarray, **kwargs: dict) -> None:
    extent = [np.min(directions), np.max(directions), 0, values.shape[0]]
    imin = np.min(values)
    imax = np.max(values)
    axes.imshow(values, norm=Normalize(vmin=imin, vmax=imax), extent=extent, **kwargs)
    axes.set_xticks(directions[::20])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title)


def build_directional_curve_display(axes: Axes, title: str,
                                    values: np.ndarray, directions: np.ndarray) -> None:
    axes.plot(directions, values)
    axes.set_xticks(directions[::20])
    plt.setp(axes.get_xticklabels(), fontsize=8)
    axes.set_title(title)


def display_initial_data(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    plt.close('all')
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.suptitle(get_display_title(local_estimator), fontsize=12)
    arrows = [(wfe.direction, wfe.energy_ratio) for wfe in local_estimator.bathymetry_estimations]
    build_image_display(axs[0, 0], 'first image original',
                        local_estimator.images_sequence[0].original_pixels,
                        directions=arrows, cmap='gray')
    build_image_display(axs[1, 0], 'second image original',
                        local_estimator.images_sequence[1].original_pixels,
                        directions=arrows, cmap='gray')
    build_image_display(axs[0, 1], 'first image filtered',
                        local_estimator.images_sequence[0].pixels,
                        directions=arrows, cmap='gray')
    build_image_display(axs[1, 1], 'second image filtered',
                        local_estimator.images_sequence[1].pixels,
                        directions=arrows, cmap='gray')
    first_radon_transform = local_estimator.radon_transforms[0]
    second_radon_transform = local_estimator.radon_transforms[1]

    values, directions = first_radon_transform.get_as_arrays()
    build_directional_2d_display(axs[0, 2], 'first radon transform', values, directions)
    values, directions = second_radon_transform.get_as_arrays()
    build_directional_2d_display(axs[1, 2], 'second radon transform', values, directions)
    plt.show()


def display_radon_transforms(local_estimator: 'SpatialDFTBathyEstimator',
                             refinement_phase: bool=False) -> None:
    plt.close('all')
    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(12, 8))
    fig.suptitle(get_display_title(local_estimator), fontsize=12)
    build_radon_transform_display(axs[:, 0], local_estimator.radon_transforms[0],
                                  'first radon transform', refinement_phase)
    build_radon_transform_display(axs[:, 2], local_estimator.radon_transforms[1],
                                  'second radon transform', refinement_phase)
    build_correl_spectrum_display(axs[:, 1], local_estimator,
                                  'Cross correlation spectrum', refinement_phase)
    plt.show()


def build_radon_transform_display(axs: Axes, transform: WavesRadon, title: str,
                                  refinement_phase: bool=False) -> None:
    values, directions = transform.get_as_arrays()
    sino_fft = transform.get_sinograms_standard_dfts()
    dft_amplitudes = np.abs(sino_fft)
    dft_phases = np.angle(sino_fft)
    variances = transform.get_sinograms_variances()
    energies = transform.get_sinograms_energies()

    build_directional_2d_display(axs[0], title, values, directions, aspect='auto', cmap='gray')
    build_directional_2d_display(axs[1], 'Sinograms DFT amplitude', dft_amplitudes, directions)
    build_directional_2d_display(axs[2], 'Sinograms DFT phase', dft_phases, directions, cmap='hsv')

    build_directional_curve_display(axs[3], 'Sinograms Variances / Energies', variances, directions)


def build_correl_spectrum_display(axs: Axes, local_estimator: 'SpatialDFTBathyEstimator',
                                  title: str, refinement_phase: bool) -> None:
    radon_transform = local_estimator.radon_transforms[0]
    if not refinement_phase:
        _, directions = radon_transform.get_as_arrays()
    else:
        directions = radon_transform.directions_interpolated_dft
    metrics = local_estimator.metrics
    key = 'interpolated_dft' if refinement_phase else 'standard_dft'
    sinograms_correlation_fft = metrics[key]['sinograms_correlation_fft']
    total_spectrum = metrics[key]['total_spectrum']
    total_spectrum_normalized = metrics[key]['total_spectrum_normalized']
    max_heta = metrics[key]['max_heta']

    build_directional_2d_display(axs[1], 'Sinograms correlation DFT module',
                                 np.abs(sinograms_correlation_fft), directions)
    build_directional_2d_display(axs[2], 'Sinograms correlation DFT Phase',
                                 np.angle(sinograms_correlation_fft), directions)
    build_directional_2d_display(axs[4], 'Sinograms correlation total spectrum',
                                 total_spectrum, directions)
    build_directional_curve_display(axs[5], 'Sinograms correlation total spectrum normalized',
                                    total_spectrum_normalized, directions)


def display_energies(local_estimator: 'SpatialDFTBathyEstimator',
                     radon1_obj: WavesRadon, radon2_obj: WavesRadon) -> None:
    fig, ax = plt.subplots()
    fig.suptitle(get_display_title(local_estimator), fontsize=12)

    image1_energy = local_estimator.images_sequence[0].energy_inner_disk
    image2_energy = local_estimator.images_sequence[1].energy_inner_disk
    ax.plot(radon1_obj.get_sinograms_energies() / image1_energy)
    ax.plot(radon2_obj.get_sinograms_energies() / image2_energy)
    plt.show()


def animate_sinograms(local_estimator: 'SpatialDFTBathyEstimator',
                      radon1_obj: WavesRadon, radon2_obj: WavesRadon) -> None:

    fig, ax = plt.subplots()
    fig.suptitle(get_display_title(local_estimator), fontsize=12)

    sinogram1_init = radon1_obj[radon1_obj.directions[0]]
    sinogram2_init = radon2_obj[radon2_obj.directions[0]]
    image1_energy = local_estimator.images_sequence[0].energy_inner_disk
    image2_energy = local_estimator.images_sequence[0].energy_inner_disk

    line1, = ax.plot(sinogram1_init.values)
    line2, = ax.plot(sinogram2_init.values)
    values1, _ = radon1_obj.get_as_arrays()
    min_radon = np.amin(values1)
    max_radon = np.amax(values1)
    plt.ylim(min_radon, max_radon)
    dir_text = ax.text(0, max_radon * 0.9, f'direction: 0, energy1: {sinogram1_init.energy}',
                       fontsize=10)

    def animate(direction: float):
        sinogram1 = radon1_obj[direction]
        sinogram2 = radon2_obj[direction]
        line1.set_ydata(sinogram1.values)  # update the data.
        line2.set_ydata(sinogram2.values)  # update the data.
        dir_text.set_text(f'direction: {direction:4.1f}, '
                          f' energy1: {sinogram1.energy/image1_energy:3.1f}, '
                          f'energy2: {sinogram2.energy/image2_energy:3.1f}')
        return line1, line2, dir_text

    ani = animation.FuncAnimation(
        fig, animate, frames=radon1_obj.directions, interval=100, blit=True, save_count=50)
    plt.show()


def display_context(local_estimator: 'SpatialDFTBathyEstimator') -> None:
    radon1 = local_estimator.radon_transforms[0]
    radon2 = local_estimator.radon_transforms[1]

    plt.close('all')
    values1, _ = radon1.get_as_arrays()
    values2, _ = radon2.get_as_arrays()
    delta_radon = np.abs(values1 - values2)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.suptitle(get_display_title(local_estimator), fontsize=12)
    axs[0, 0].imshow(radon1.pixels, aspect='auto', cmap='gray')
    axs[0, 0].set_title('subI_Det0')
    axs[1, 0].imshow(values1, aspect='auto', cmap='gray')
    axs[1, 0].set_title('radon image1')
    axs[0, 1].imshow(radon2.pixels, aspect='auto', cmap='gray')
    axs[0, 1].set_title('subI_Det1')
    axs[1, 1].imshow(values2, aspect='auto', cmap='gray')
    axs[1, 1].set_title('radon image2')
    sinograms1_energies = radon1.get_sinograms_energies()
    sinograms2_energies = radon2.get_sinograms_energies()
    image1_energy = local_estimator.images_sequence[0].energy_inner_disk
    image2_energy = local_estimator.images_sequence[1].energy_inner_disk
    axs[0, 2].plot(sinograms1_energies / image1_energy)
    axs[0, 2].plot(sinograms2_energies / image2_energy)
    axs[0, 2].set_title('directions energies')
    axs[1, 2].imshow(delta_radon, aspect='auto', cmap='gray')
    axs[1, 2].set_title('radon1 - radon2')
    plt.show()
    display_energies(local_estimator, radon1, radon2)
    animate_sinograms(local_estimator, radon1, radon2)
