# -*- coding: utf-8 -*-
""" BathyLauncher main file - API definition

:author: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 2021
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
import cProfile
from datetime import datetime
from pathlib import Path
from shutil import copy
import time
from typing import Type  # @NoMove

import click
import yaml

from s2shores.image.ortho_stack import OrthoStack  # @NoMove

from s2shores.bathylauncher.bathy_launcher import BathyLauncher, ProductsDescriptor  # @UnusedImport
from s2shores.bathylauncher.products.geotiff_product import GeoTiffProduct
from s2shores.bathylauncher.products.s2_image_product import S2ImageProduct


def create_timestamped_dir(output_dir_root: Path) -> Path:
    """
    Creates subdirectories in output_dir_root.
    """
    date = datetime.now().strftime('%Y_%m_%d_%H-%M-%S')

    output_dir_step = output_dir_root / f'run_{date}'
    output_dir_step.mkdir(parents=True, exist_ok=False)
    return output_dir_step

def bathy_process(input_product: Path, product_type: str, output_dir: Path,
                    config_file: Path, debug_file: Path = None, debug_path: Path = None, distoshore_file: Path = None,
                    delta_times_dir: Path = None, roi_file: Path = None, limit_to_roi : bool = False,
                    nb_subtiles: int = 1, sequential: bool  = False, profiling: bool = False):
    """ Function to be called when using bathylauncher as a module """
    product_cls: Type[OrthoStack]
    if product_type == 'geotiff':
        product_cls = GeoTiffProduct
    elif product_type == 'S2':
        product_cls = S2ImageProduct
    else:
        raise TypeError('Product type not handled')
    with open(config_file) as file:
        bathy_inversion_params = yaml.load(file, Loader=yaml.FullLoader)

    output_dir_processing = create_timestamped_dir(output_dir)
    copy(config_file, output_dir_processing)

    debug_params = None
    if debug_file:
        with open(debug_file) as file:
            debug_params = yaml.load(file, Loader=yaml.FullLoader)
        if not debug_path:
            debug_path = output_dir_processing / 'debug'
            debug_path.mkdir()
        else:
            if not debug_path.exists():
                raise FileExistsError

    products: ProductsDescriptor = {input_product.stem:  # pylint: disable=possibly-unused-variable
                                        (input_product, product_cls, output_dir_processing,
                                         bathy_inversion_params, nb_subtiles, delta_times_dir,
                                         distoshore_file, roi_file, limit_to_roi,
                                         debug_params, debug_path)
                                    }
    start = time.time()

    # In order to be able to do profiling, the launch() calling instruction must be a string.
    # Therefore, in order to avoid duplication, we will be calling launch() with eval when no
    # profiling is required.
    launch_instruction = \
        'BathyLauncher.launch(products,' \
        'gravity_type="LATITUDE_VARYING",sequential_run=sequential)'
    if profiling:
        sequential = True
        cProfile.runctx(launch_instruction, globals(), locals())
    else:
        BathyLauncher.launch(products, gravity_type="LATITUDE_VARYING", sequential_run=sequential)  # pylint: disable=eval-used

    stop = time.time()
    print('Bathy estimation total time : ', stop - start)

@click.command()
@click.option('--input_product', type=click.Path(exists=True, path_type=Path), required=True,
              help='Path to input product')
@click.option('--product_type',
              type=click.Choice(['S2', 'geotiff'], case_sensitive=False), required=True)
@click.option('--output_dir', type=click.Path(exists=True, path_type=Path), required=True,
              help='Output directory.')
@click.option('--config_file', type=click.Path(exists=True, path_type=Path), required=True,
              help='YAML config file for bathymetry computation')
@click.option('--debug_file', type=click.Path(exists=True, path_type=Path), required=False,
              help='YAML config file for bathymetry debug definition')
@click.option('--debug_path', type=click.Path(exists=True, path_type=Path), required=False,
              help='path to store debug information')
@click.option('--distoshore_file', type=click.Path(exists=True, path_type=Path), required=False,
              help='georeferenced netCDF file giving the distance of a point to the closest shore')
@click.option('--delta_times_dir', type=click.Path(exists=True, path_type=Path), required=False,
              help='Directory containing the files describing S2A and S2B delta times between '
              'detectors. Mandatory for processing a Sentinel2 product.')
@click.option('--roi_file', type=click.Path(exists=True, path_type=Path), required=False,
              help='vector file specifying the polygon(s) where the bathymetry must be computed')
@click.option('--limit_to_roi', is_flag=True,
              help='if set and roi_file is specified, limit the bathymetry output to that roi')
@click.option('--nb_subtiles', default=1, help='Number of subtiles')
@click.option('--sequential/--no-sequential', default=False,
              help='if set, allows run in a single thread, usefull for debugging purpose')
@click.option('--profiling/--no-profiling', default=False,
              help='If set, print profiling information about the whole bathymetry estimation')
def process_command(input_product: Path, product_type: str, output_dir: Path,
                    config_file: Path, debug_file: Path, debug_path: Path, distoshore_file: Path,
                    delta_times_dir: Path, roi_file: Path, limit_to_roi: bool,
                    nb_subtiles: int, sequential: bool, profiling: bool) -> None:
    bathy_process(input_product, product_type, output_dir,
                    config_file, debug_file, debug_path, distoshore_file,
                    delta_times_dir, roi_file, limit_to_roi,
                    nb_subtiles, sequential, profiling)



if __name__ == '__main__':
    process_command() # pylint: disable=no-value-for-parameter
