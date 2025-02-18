# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for s2shores.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
from dataclasses import dataclass

import pytest
from pathlib import Path

@dataclass
class S2SHORESTestsPath():
    project_dir: Path
    def __post_init__(self):
        s2new = "S2A_MSIL1C_20200622T105631_N0500_R094_T30TXR_20231110T094313.SAFE"
        s2old = "S2A_MSIL1C_20200622T105631_N0209_R094_T30TXR_20200622T130553.SAFE"

        self.config_dir = self.project_dir / "TestsS2Shores" / "config"
        self.yaml_file = "wave_bathy_inversion_config.yaml"
        self.output_dir = self.project_dir / "TestsS2Shores" / "output"
        self.cli_path = self.project_dir / "src" / "s2shores" / "bathylauncher" / "bathy_processing.py"
        self.s2new_product_dir = (self.project_dir / "TestsS2Shores" / "products" / "S2_30TXR_NEW"
                                  / s2new)
        self.s2old_product_dir = (self.project_dir / "TestsS2Shores" / "products" / "S2_30TXR_OLD"
                                  / s2old)
        self.pneo_product_dir = (self.project_dir / "TestsS2Shores" / "products" /
                                 "PNEO_DUCK" / "Duck_PNEO_XS_b3_VT.tif")
        self.funwave_product_dir = (self.project_dir / "TestsS2Shores" / "products" /
                                    "FUNWAVE" / "funwave_cropped.tif")
        self.swach7_product_dir = (self.project_dir / "TestsS2Shores" / "products" /
                                   "SWASH_7_4" / "testcase_7_4_cropped.tif")
        self.swach8_product_dir = (self.project_dir / "TestsS2Shores" / "products" /
                                   "SWASH_8_2" / "testcase_8_2.tif")
        self.delta_times_dir = self.project_dir / "src" / "s2shores" / "bathylauncher" / "config"
        self.dis2shore_dir = self.project_dir / "TestsS2Shores" / "distoshore"
        self.debug_dir = self.project_dir / "TestsS2Shores" / "debug"
        self.roi_dir = self.project_dir / "TestsS2Shores" / "ROI"

@pytest.fixture
def s2shores_paths(request) -> S2SHORESTestsPath:
    return S2SHORESTestsPath(Path(__file__).parent.parent)
