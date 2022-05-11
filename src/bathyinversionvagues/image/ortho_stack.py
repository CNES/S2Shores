# -*- coding: utf-8 -*-
""" Definition of the OrthoStack class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List  # @NoMove

from osgeo import gdal
from shapely.geometry import Polygon

from ..data_providers.delta_time_provider import DeltaTimeProvider
from ..image_processing.waves_image import WavesImage
from .image_geometry_types import MarginsType
from .ortho_layout import OrthoLayout
from .ortho_sequence import FrameIdType, FramesIdsType
from .sampled_ortho_image import SampledOrthoImage


class OrthoStack(ABC, OrthoLayout):
    """ An orthorectified stack is a set of images also called frames which have the following
    characteristics :

    - all the frames are orthorectified in the same cartographic system
    - they have been acquired by the same sensor, almost at the time. The maximum delay between the
      first and the last acquisition is typically of few minutes.
    - they have the same footprint as well as the same resolution
    - thus they have the same size in pixels
    - the frames can be located in a single file or in several distinct files, possibly spread in
      different locations.
    - when several images are contained in a product or in a directory not all of them are
      considered as frames of the OrthoStack. Just a subset of them are declared as frames, which
      allows for instance to select images of the same resolution from the set of images.
    """

    def __init__(self, product_path: Path) -> None:
        """ Constructor.

        :param product_path: Path to the file or directory corresponding to this ortho stack
        """
        self._product_path = product_path

        # Extract the relevant information from the first usable spectral band
        # FIXME: use the selected frames instead ?
        im_dataset = gdal.Open(str(self.get_image_file_path(self.usable_frames[0])))

        super().__init__(im_dataset.RasterXSize, im_dataset.RasterYSize,
                         im_dataset.GetProjection(), im_dataset.GetGeoTransform())
        # We are done with info retrieval: release the dataset
        im_dataset = None

    @property
    def product_path(self) -> Path:
        """ Path to this product
        """
        return self._product_path

    @property
    @abstractmethod
    def full_name(self) -> str:
        """ :returns: the full name of this ortho stack
        """

    @property
    @abstractmethod
    def short_name(self) -> str:
        """ :returns: the short name of the orthorectified stack
        """

    @property
    @abstractmethod
    def satellite(self) -> str:
        """ :returns: the satellite identifier which acquired the frames
        """

    @property
    @abstractmethod
    def acquisition_time(self) -> str:
        """ :returns: the approximate acquisition time of the stack. Typically the central frame
        acquisition date and time.
        """

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this ortho stack
        """
        infos = {
            'sat': self.satellite,  #
            'AcquisitionTime': self.acquisition_time,  #
            'epsg': 'EPSG:' + str(self.epsg_code)}
        return infos

    @property
    @abstractmethod
    def usable_frames(self) -> FramesIdsType:
        """ :returns: the list of identifiers of the frames which can be used in the stack.
                      This can be a subset of all the available frames in the stack, for instance
                      spectral bands at the same resolution or acquisitions made at consistent
                      times.
        """

    @abstractmethod
    def get_image_file_path(self, frame_id: FrameIdType) -> Path:
        """ Provides the full path to the file containing a given frame of this ortho stack

        :param frame_id: the identifier of the frame (e.g. 'B02', or 2, or a datetime)
        :returns: the path to the file containing the frame pixels
        """

    @abstractmethod
    def get_frame_index_in_file(self, frame_id: FrameIdType) -> int:
        """ Provides the index of a given frame of this ortho stack in the file specified
        by get_image_file_path()

        :param frame_id: the identifier of the frame (e.g. 'B02', or 2, or a datetime)
        :returns: the index of the layer in the file where the frame pixels are contained
        """

    @abstractmethod
    def create_delta_time_provider(
            self, external_delta_times_path: Optional[Path] = None) -> DeltaTimeProvider:
        """ Build and returns a DeltaTimeProvider suitable for this OrthoStack. It may be built
        using only data contained inside the ortho stack, or it may need to use data from a file
        which is external to the orhto stack.

        :param external_delta_times_path: path to a file or a directory containing data necessary
                                          to build the DeltaTimeProvider when they are not inside
                                          the ortho stack itself.
        :returns: a DeltaTimeProvider fully configured for being used with this ortho stack.
        """

    def build_subtiles(self, nb_subtiles_max: int, step_x: float, step_y: float,
                       margins: MarginsType, roi: Optional[Polygon] = None) \
            -> List['SampledOrthoImage']:
        """ Class method building a set of SampledOrthoImage instances, forming a tiling of the
        specified orthorectifed image.

        :param image: the orthorectified image onto which the sampling is defined
        :param nb_subtiles_max: the meximum number of tiles to create
        :param step_x: the sampling step to use along the X axis for building the tiles
        :param step_y: the sampling step to use along the X axis for building the tiles
        :param margins: the margins to consider around the samples to determine the image extent
        :param roi: theroi for which bathymetry must be computed, if any.
        :returns: a list of SampledOrthoImage objects covering the orthorectfied image with the
                  specified sampling steps and margins.
        """
        ortho_sampling = self.get_samples_positions(step_x, step_y, margins, roi)
        subtiles_samplings = ortho_sampling.split(nb_subtiles_max)
        subtiles: List[SampledOrthoImage] = []
        for subtile_sampling in subtiles_samplings:
            subtiles.append(SampledOrthoImage(self, subtile_sampling, margins))
        return subtiles

    def read_frame_image(self, frame_id: FrameIdType, line_start: int, line_stop: int,
                         col_start: int, col_stop: int) -> WavesImage:
        """ Read a rectangle of pixels from a specific frame of this stack.

        :param frame_id: the identifier of the  frame to read
        :param line_start: the image line where the rectangle begins
        :param line_stop: the image line where the rectangle stops
        :param col_start: the image column where the rectangle begins
        :param col_stop: the image column where the rectangle stops
        :returns: a sub image taken from the frame
        """
        image_dataset = gdal.Open(str(self.get_image_file_path(frame_id)))
        image = image_dataset.GetRasterBand(self.get_frame_index_in_file(frame_id))
        nb_cols = col_stop - col_start + 1
        nb_lines = line_stop - line_start + 1
        pixels = image.ReadAsArray(col_start, line_start, nb_cols, nb_lines)
        # release dataset
        image_dataset = None
        return WavesImage(pixels, self._geo_transform.resolution)
