# -*- coding: utf-8 -*-
"""
Tests to ensure no code regression, the outputs are compared to reference results.


:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:license: see LICENSE file
:created: 06/03/2025

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import os
import shutil

from osgeo import gdal
from tests.test_utils import compare_files, unzip_file
from src.s2shores.bathylauncher.bathy_processing import bathy_process
import pytest
from tests.conftest import S2SHORESTestsPath


@pytest.mark.ci
def test_nominal_spatialCorrelation_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and global distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = "GMT_intermediate_coast_distance_01d_test_5000_cropped.nc"

    bathy_process(
        input_product=s2shores_paths.s2new_cropped,
        product_type='S2',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config2"/"wave_bathy_inversion_config_quick.yaml"),
        distoshore_file=(s2shores_paths.dis2shore_dir / dis2shore_file),
        delta_times_dir=s2shores_paths.delta_times_dir_esa,
        nb_subtiles=36,
    )

    compare_files(reference_dir = f"{s2shores_paths.output_dir}/CI_tests/nominal_spatialCorrelation_s2_quick",
                  output_dir = s2shores_paths.output_dir)

@pytest.mark.ci
def test_nominal_dft_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and tile distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    bathy_process(
        input_product=s2shores_paths.s2new_cropped,
        product_type='S2',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config3"/"wave_bathy_inversion_config_quick.yaml"),
        debug_file=None,
        debug_path=None,
        distoshore_file=(s2shores_paths.dis2shore_dir/"disToShore_30TXR_cropped.TIF"),
        delta_times_dir=s2shores_paths.delta_times_dir_esa,
        nb_subtiles=36,
    )

    compare_files(reference_dir = f"{s2shores_paths.output_dir}/CI_tests/nominal_dft_s2_quick",
                  output_dir = s2shores_paths.output_dir)

@pytest.mark.ci
def test_nominal_video_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data without ROI and distoshore, with
    geotiff product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    unzip_file(s2shores_paths.funwave_cropped.with_suffix('.zip'))

    bathy_process(
        input_product=s2shores_paths.funwave_cropped,
        product_type='geotiff',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config4"/"wave_bathy_inversion_config_quick.yaml"),
        nb_subtiles=4,
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/nominal_video_quick",
                  output_dir=s2shores_paths.output_dir)

@pytest.mark.ci
def test_debug_pointswash_temporal_corr_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH7.4 data without ROI, with geotiff product, temporal
    correlation debug, grid debug point mode and Layers-type expert.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    debug_path = f'{s2shores_paths.output_dir}/debug_pointswash_temporal_corr'
    if os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    unzip_file(s2shores_paths.swash7_cropped.with_suffix('.zip'))

    bathy_process(
        input_product=s2shores_paths.swash7_cropped,
        product_type='geotiff',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config7"/"wave_bathy_inversion_config_quick.yaml"),
        debug_file=(s2shores_paths.debug_dir/"debug_points_SWASH_7_4_cropped.yaml"),
        debug_path=(s2shores_paths.output_dir/"debug_pointswash_temporal_corr"),
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswash_temporal_corr_quick",
                  output_dir=s2shores_paths.output_dir,
                  debug_dir = f'{s2shores_paths.output_dir}/debug_pointswash_temporal_corr')

@pytest.mark.ci
def test_debug_pointswash_spatial_dft_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial dft debug and grid debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    debug_path = f'{s2shores_paths.output_dir}/debug_pointswash_spatial_dft'
    if os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    unzip_file(s2shores_paths.swash8_cropped.with_suffix('.zip'))

    bathy_process(
        input_product=s2shores_paths.swash8_cropped,
        product_type='geotiff',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config5"/"wave_bathy_inversion_config_quick.yaml"),
        debug_file=(s2shores_paths.debug_dir/"debug_points_SWASH_8_2_cropped.yaml"),
        debug_path=(s2shores_paths.output_dir/"debug_pointswash_spatial_dft")
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswash_dft_quick",
                  output_dir=s2shores_paths.output_dir,
                  debug_dir = f'{s2shores_paths.output_dir}/debug_pointswash_spatial_dft')

@pytest.mark.ci
def test_debug_pointswash_spatial_corr_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial
    correlation debug, grid debug point mode and Layers-type nominal.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    debug_path = f'{s2shores_paths.output_dir}/debug_pointswash_spatial_corr'
    if os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    unzip_file(s2shores_paths.swash8_cropped.with_suffix('.zip'))

    bathy_process(
        input_product=s2shores_paths.swash8_cropped,
        product_type='geotiff',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config6"/"wave_bathy_inversion_config_quick.yaml"),
        debug_file=(s2shores_paths.debug_dir/"debug_points_SWASH_8_2_cropped.yaml"),
        debug_path=(s2shores_paths.output_dir/"debug_pointswash_spatial_corr")
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_pointswash_spatial_corr_quick",
                  output_dir=s2shores_paths.output_dir,
                  debug_dir = f'{s2shores_paths.output_dir}/debug_pointswash_spatial_corr')

@pytest.mark.ci
def test_limitroi_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data with ROI, ROI limit and sequential option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    bathy_process(
        input_product=s2shores_paths.s2new_cropped,
        product_type='S2',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config2"/"wave_bathy_inversion_config_quick.yaml"),
        distoshore_file=(s2shores_paths.dis2shore_dir/"disToShore_30TXR_cropped.TIF"),
        delta_times_dir=s2shores_paths.delta_times_dir_esa,
        roi_file=(s2shores_paths.roi_dir/"30TXR-ROI-cropped.shp"),
        limit_to_roi=True,
        nb_subtiles=36,
        sequential=True,
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/limitroi_s2_quick",
                  output_dir=s2shores_paths.output_dir)

@pytest.mark.ci
def test_debug_mode_point_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data with S2 product and point debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    debug_path = f'{s2shores_paths.output_dir}/debug_mode_point_s2'
    if os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    bathy_process(
        input_product=s2shores_paths.s2new_cropped,
        product_type='S2',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config8"/"wave_bathy_inversion_config_quick.yaml"),
        debug_file=(s2shores_paths.debug_dir/"debug_points_30TXR_notongrid_cropped.yaml"),
        debug_path=(s2shores_paths.output_dir/"debug_mode_point_s2"),
        distoshore_file=(s2shores_paths.dis2shore_dir/"disToShore_30TXR_cropped.TIF"),
        delta_times_dir=s2shores_paths.delta_times_dir_esa,
        nb_subtiles=36,
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_mode_point_s2_quick",
                  output_dir=s2shores_paths.output_dir,
                  debug_dir = f'{s2shores_paths.output_dir}/debug_mode_point_s2')


@pytest.mark.ci
def test_debug_area_funwave_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data with geotiff product and debug area.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    debug_path = f'{s2shores_paths.output_dir}/debug_area_funwave'
    if os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.mkdir(debug_path)

    unzip_file(s2shores_paths.funwave_cropped.with_suffix('.zip'))

    bathy_process(
        input_product=s2shores_paths.funwave_cropped,
        product_type='geotiff',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config9"/"wave_bathy_inversion_config_quick.yaml"),
        debug_file=(s2shores_paths.debug_dir/"debug_area_funwave_cropped.yaml"),
        debug_path=(s2shores_paths.output_dir/"debug_area_funwave")
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/debug_area_funwave_quick",
                  output_dir=s2shores_paths.output_dir,
                  debug_dir = f'{s2shores_paths.output_dir}/debug_area_funwave')

@pytest.mark.ci
def test_roi_profiling_s2_quick(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI limit
    , with S2 product, ROI and profiling option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = ("GMT_intermediate_coast_distance_01d_test_5000_cropped.nc")

    bathy_process(
        input_product=s2shores_paths.s2new_cropped,
        product_type='S2',
        output_dir=s2shores_paths.output_dir,
        config_file=(s2shores_paths.config_dir/"config2"/"wave_bathy_inversion_config_quick.yaml"),
        distoshore_file=(s2shores_paths.dis2shore_dir/dis2shore_file),
        delta_times_dir=s2shores_paths.delta_times_dir_esa,
        roi_file=(s2shores_paths.roi_dir/"30TXR-ROI-cropped.shp"),
        nb_subtiles=36,
        profiling=True
    )

    compare_files(reference_dir=f"{s2shores_paths.output_dir}/CI_tests/roi_profiling_s2_quick",
                  output_dir=s2shores_paths.output_dir)
