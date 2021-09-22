# -*- coding: utf-8 -*-
""" Definition of the OrthoStack class

:author: GIROS Alain
:created: 17/05/2021
"""
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from osgeo import gdal

from ..image_processing.waves_image import WavesImage
from .ortho_layout import OrthoLayout
from typing import Dict, Union  # @NoMove


FrameIdType = Union[str, int, datetime]


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
      allows for instance to select images of the same resolution to build the stack from
      the set of images..
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

    @property
    def spatial_resolution(self) -> float:
        """ :returns: the spatial resolution of the different frames in the stack (m)
        """
        return self._geo_transform.resolution

    @property
    def x_y_resolutions_equal(self) -> bool:
        """ :returns: True if the absolute values of X and Y resolutions of the frames are equal
        """
        return self._geo_transform.x_y_resolutions_equal

    def build_infos(self) -> Dict[str, str]:
        """ :returns: a dictionary of metadata describing this ortho stack
        """
        infos = {
            'sat': self.satellite,  #
            'AcquisitionTime': self.acquisition_time,  #
            'epsg': 'EPSG:' + str(self.epsg_code)}
        return infos

    @abstractmethod
    def get_image_file_path(self, frame_id: FrameIdType) -> Path:
        """ Provides the full path to the file containing a given frame of this ortho stack

        :param frame_id: the identifier of the frame (e.g. 'B02', or 2, or a datetime)
        :returns: the path to the file containing the frame pixels
        """

    @abstractmethod
    def get_band_index_in_file(self, frame_id: FrameIdType) -> int:
        """ Provides the index of a given frame of this ortho stack in the file specified
        by get_image_file_path()

        :param frame_id: the identifier of the frame (e.g. 'B02', or 2, or a datetime)
        :returns: the index of the layer in the file where the frame pixels are contained
        """

    def read_pixels(self, frame_id: FrameIdType, line_start: int, line_stop: int,
                    col_start: int, col_stop: int) -> WavesImage:
        """ Read a rectangle of pixels from a specific frame of this image.

        :param frame_id: the identifier of the  frame to read
        :param line_start: the image line where the rectangle begins
        :param line_stop: the image line where the rectangle stops
        :param col_start: the image column where the rectangle begins
        :param col_stop: the image column where the rectangle stops
        :returns: a sub image taken from the frame
        """
        image_dataset = gdal.Open(str(self.get_image_file_path(frame_id)))
        image = image_dataset.GetRasterBand(self.get_band_index_in_file(frame_id))
        nb_cols = col_stop - col_start + 1
        nb_lines = line_stop - line_start + 1
        pixels = image.ReadAsArray(col_start, line_start, nb_cols, nb_lines)
        # release dataset
        image_dataset = None
        return WavesImage(pixels, self.spatial_resolution)
