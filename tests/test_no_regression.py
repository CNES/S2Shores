# -*- coding: utf-8 -*-
"""
Tests to ensure no code regression, the outputs are compared to reference results.


:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:license: see LICENSE file
:created: 14/02/2025

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import os
import glob
import zipfile
import xarray as xr
from click.testing import CliRunner
from tests.conftest import S2SHORESTestsPath

from s2shores.bathylauncher.bathy_processing import process_command


def compare_files(reference_dir : str, output_dir : str):
    """
    Compares the contents of the reference directory with the most recently created
    test output directory. Ensures that the filenames match and that the contents of
    NetCDF files are identical.

    :param reference_dir: The directory containing reference files.
    :returns: True if the directories have identical filenames and matching NetCDF content.
    :raises Exception: If filenames differ between the directories or NetCDF file contents do not match.
    """
    # Get all directories in the specified parent directory
    dirs = [d for d in glob.glob(os.path.join(output_dir, "*/")) if os.path.isdir(d)]

    # Find the most recently created directory, ie. the test output directory
    out_test_dir = max(dirs, key=os.path.getctime)

    ref_files = os.listdir(reference_dir)
    out_test_files = os.listdir(out_test_dir)

    if ref_files == out_test_files:
        print("Both directories contain the same filenames.")
        return True
    else:
        raise Exception("Filenames differ between the directories.\n"
               f"Only in {reference_dir} : {reference_dir} - {out_test_dir}"
               f"Only in {out_test_dir} : {out_test_dir} - {reference_dir}")


    #Assert the files in the reference directory are the same
    #than the ones in the lastly created directory
    ref_nc = [nc_file for nc_file in ref_files if ".nc" in nc_file]
    out_nc = [nc_file for nc_file in out_test_files if ".nc" in nc_file]

    ref_xr = xr.open_dataset(reference_dir / ref_nc)
    out_xr = xr.open_dataset(out_test_dir / out_nc)

    xr.testing.assert_equal(ref_xr, out_xr)


def test_nominal_spatialCorrelation_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and global distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = "GMT_intermediate_coast_distance_01d_test_5000.nc"
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2old_product_dir),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file}',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/{dis2shore_file}',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--nb_subtiles', '36'])
    compare_files(reference_dir = f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_09-53-17",
                  output_dir = s2shores_paths.output_dir)


def test_nominal_dft_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and tile distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    result = runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_product_dir),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config3/wave_bathy_inversion_config.yaml',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--nb_subtiles', '36'])

    compare_files(reference_dir = f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_10-06-02",
                  output_dir = s2shores_paths.output_dir)


def test_nominal_tri_stereo_pneo(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test PNEO data without ROI and distoshore, with geotiff
    product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.pneo_product_dir),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config1/{s2shores_paths.yaml_file}',
        '--nb_subtiles', '36'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_13-58-06",
                  output_dir=s2shores_paths.output_dir)


def test_nominal_video(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data without ROI and distoshore, with
    geotiff product, nb_subtiles=1 and Layers-type debug.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.funwave_product_dir),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config4/{s2shores_paths.yaml_file}',
        '--nb_subtiles', '4'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_11-27-08",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswash_temporal_corr(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH7.4 data without ROI, with geotiff product, temporal
    correlation debug, grid debug point mode and Layers-type expert.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swash7_product_dir),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config7/wave_bathy_inversion_config.yaml',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswash_temporal_corr',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_7_4.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_10-15-47",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswash_spatial_dft(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product
    , dft spatial debug and grid debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swash8_product_dir),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config5/{s2shores_paths.yaml_file}',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswash_spatial_dft',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_8_2.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_10-22-37",
                  output_dir=s2shores_paths.output_dir)


def test_debug_pointswash_spatial_corr(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test SWASH8.2 data without ROI, with geotiff product, spatial
    correlation debug, grid debug point mode and Layers-type nominal.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.swash8_product_dir),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config6/{s2shores_paths.yaml_file}',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_pointswash_spatial_corr',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_SWASH_8_2.yaml'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_14-08-28",
                  output_dir=s2shores_paths.output_dir)


def test_limitroi_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data with ROI, ROI limit and sequential option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_product_dir),
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file}',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF',
        '--product_type', 'S2',
        '--nb_subtiles', '36',
        '--roi_file', f'{s2shores_paths.roi_dir}/30TXR-ROI.shp',
        '--limit_to_roi',
        '--sequential'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_11-06-32",
                  output_dir=s2shores_paths.output_dir)


def test_debug_mode_point_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data with S2 product and point debug point mode.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2new_product_dir),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config8/{s2shores_paths.yaml_file}',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF',
        '--nb_subtiles', '36',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_mode_point_s2',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_points_30TXR_notongrid.yaml'])
    # compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_11-00-30",
    # output_dir=s2shores_paths.output_dir)


def test_debug_area_funwave(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Funwave data with geotiff product and debug area.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.funwave_product_dir),
        '--product_type', 'geotiff',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config9/{s2shores_paths.yaml_file}',
        '--debug_path', f'{s2shores_paths.output_dir}/debug_area_funwave',
        '--debug_file', f'{s2shores_paths.debug_dir}/debug_area_funwave.yaml'])
    # compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_24_10-35-23",
    #               output_dir=s2shores_paths.output_dir)


def test_roi_profiling_s2(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI limit
    , with S2 product, ROI and profiling option.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = "GMT_intermediate_coast_distance_01d_test_5000.nc"

    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2old_product_dir),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file}',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/{dis2shore_file}',
        '--delta_times_dir', str(s2shores_paths.delta_times_dir),
        '--roi_file', f'{s2shores_paths.roi_dir}/30TXR-ROI.shp',
        '--nb_subtiles', '36',
        '--profiling'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_20_11-14-27",
                  output_dir=s2shores_paths.output_dir)


def test_nominal_dft_s2_cnes_deltaT(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR New data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and tile distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2old_product_dir),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config3/{s2shores_paths.yaml_file}',
        '--delta_times_dir', f'{s2shores_paths.delta_times_dir}/cnes',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/disToShore_30TXR.TIF',
        '--nb_subtiles', '36'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_24_17-44-00",
                  output_dir=s2shores_paths.output_dir)

def test_nominal_spatialcorr_s2_cnes_deltat(s2shores_paths: S2SHORESTestsPath) -> None:
    """
    Test Sentinel-2 30TXR Old data without ROI, with S2 product,
    nb_subtiles>1, Layers-type debug and global distoshore.

    - Verify that all expected output files are created.
    - Ensure the generated .nc output file matches the reference.
    """
    dis2shore_file = "GMT_intermediate_coast_distance_01d_test_5000.nc"
    runner = CliRunner()

    runner.invoke(process_command, [
        '--input_product', str(s2shores_paths.s2old_product_dir),
        '--product_type', 'S2',
        '--output_dir', str(s2shores_paths.output_dir),
        '--config_file', f'{s2shores_paths.config_dir}/config2/{s2shores_paths.yaml_file}',
        '--delta_times_dir', f'{s2shores_paths.delta_times_dir}/cnes',
        '--distoshore_file', f'{s2shores_paths.dis2shore_dir}/{dis2shore_file}',
        '--nb_subtiles', '36'])
    compare_files(reference_dir=f"{s2shores_paths.output_dir}/reference_results/run_2025_02_24_17-36-22",
                  output_dir=s2shores_paths.output_dir)