# -*- coding: utf-8 -*-
""" Definition of the SampledOrthoImage class

:author: GIROS Alain
:created: 05/05/2021
"""
from typing import TYPE_CHECKING  # @NoMove

from shapely.geometry import Point

from ..image_processing.waves_image import WavesImage
from .image_geometry_types import MarginsType, ImageWindowType
from .sampling_2d import Sampling2D


if TYPE_CHECKING:
    from .ortho_stack import OrthoStack, FrameIdType  # @UnusedImport


class SampledOrthoImage:
    """ This class makes the link between a Sampling2D and the image in which it is defined.
    """

    def __init__(self, ortho_stack: 'OrthoStack', carto_sampling: Sampling2D,
                 margins: MarginsType) -> None:
        """ Define the samples belonging to the subtile. These samples correspond to the cross
        product of the X and Y coordinates.

        :param ortho_stack: the orthorectified stack onto which the sampling is defined
        :param carto_sampling: the sampling of this SampledOrthoImage
        :param margins: the margins to consider around the samples to determine the image extent
        """
        self.ortho_stack = ortho_stack
        self.carto_sampling = carto_sampling
        self._margins = margins

        # col_start, line_start, nb_cols and nb_lines define the rectangle of pixels in image
        # coordinates which are just needed to process the subtile. No margins and no missing
        # lines or columns.
        self._line_start, _, self._col_start, _ = \
            self.ortho_stack.window_pixels(self.carto_sampling.upper_left_sample, self._margins)
        _, self._line_stop, _, self._col_stop = \
            self.ortho_stack.window_pixels(self.carto_sampling.lower_right_sample, self._margins)

    def read_frame_image(self, frame_id: 'FrameIdType') -> WavesImage:
        """ Read the whole rectangle of pixels corresponding to this SampledOrthoImage
        retrieved from a specific frame of the orthorectified stack.

        :param frame_id: the identifier of the frame in the stack
        :returns: the rectangle of pixels as an array
        """
        return self.ortho_stack.read_frame_image(frame_id,
                                                 self._line_start, self._line_stop,
                                                 self._col_start, self._col_stop)

    def window_extent(self, carto_point: Point) -> ImageWindowType:
        """ Given a point defined in the projected domain, computes a rectangle of pixels centered
        on the pixel containing this point and taking into account the SampledOrthoImage margins.

        :param carto_point: the center point
        :returns: the window as a tuple of four coordinates relative to line_start and col_start of
                  this SampledOrthoImage
        """
        return self.ortho_stack.window_pixels(carto_point, self._margins,
                                              self._line_start, self._col_start)

    def __str__(self) -> str:
        msg = str(self.carto_sampling)
        msg += f' C[{self._col_start}, {self._col_stop}] * L[{self._line_start}, {self._line_stop}]'
        return msg
