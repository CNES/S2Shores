# -*- coding: utf-8 -*-
""" Class encapsulating a list of superimposed images of the same size and resolution


:author: Alain Giros
:organization: CNES
:copyright: 2022 CNES. All rights reserved.
:license: see LICENSE file
:created: 7 april 2022
"""
from datetime import datetime

from shapely.geometry import Point
from typing import Tuple, List, Union


from ..data_providers.delta_time_provider import DeltaTimeProvider
from ..image_processing.waves_image import WavesImage
from ..waves_exceptions import SequenceImagesError
from .image_geometry_types import ImageWindowType


FrameIdType = Union[str, int, datetime]
FramesIdsType = Union[List[str], List[int], List[datetime]]


# FIXME: list or dict indexed by image_id ???
class OrthoSequence(list):
    """ Class encapsulating the information describing a sequence of superimposed images of same
    shape and resolution and providing operations on it.
    """

    def __init__(self, delta_time_provider: DeltaTimeProvider) -> None:
        super().__init__()

        self._delta_time_provider = delta_time_provider

        self._images_id: List[FrameIdType] = []
        self._images_time: List[datetime] = []

    @property
    def shape(self) -> Tuple[int, ...]:
        """ :returns: The shape common to all the images contained in this sequence of images
        :raises AttributeError: when there is no image in the sequence
        """
        if not self:
            msg = 'Shape undefined when there is no image in the OrthoSequence'
            raise AttributeError(msg)
        return self[0].pixels.shape

    @property
    def resolution(self) -> float:
        """ :returns: The spatial resolution of this sequence of images (m)
        :raises AttributeError: when there is no image in the sequence
        """
        if not self:
            msg = 'Resolution undefined when there is no image in the OrthoSequence'
            raise AttributeError(msg)
        return self[0].resolution

    def get_time_difference(self, location: Point,
                            start_frame_id: FrameIdType, stop_frame_id: FrameIdType) -> float:
        """ :returns: The time duration between the start and stop images used for the estimation.
                      Positive or negative depending on the chronology of start and stop images.
        :raises SequenceImagesError: if the start or stop frames are unknown.
        """
        if start_frame_id not in self._images_id or stop_frame_id not in self._images_id:
            msg = f'Start and/or stop frames are unknown in this image sequence: {self._images_id}'
            raise SequenceImagesError(msg)
        return self._delta_time_provider.get_delta_time(start_frame_id, stop_frame_id, location)

    def append_image(self, image: WavesImage, image_id: FrameIdType) -> None:
        """ Append a new image to this image sequence. The first appended image fixes the spatial
        resolution and the shape of all the image which will be entered in the sequence.

        :param image: the image to append at the last element of the sequence
        :param image_id: the identifier of the image, must be unique in the sequence.
        :raises ValueError: when the image has not the same shape or resolution than the images
                            already recorded or when the image identifier is already present in the
                            sequence.
        """
        if self and image.resolution != self[0].resolution:
            msg = 'Trying to add an image into images sequence with incompatible resolution:  new '
            msg += f'image resolution: {image.resolution} sequence resolution: {self.resolution}'
            raise ValueError(msg)
        if self and image.pixels.shape != self[0].pixels.shape:
            msg = 'Trying to add an image into images sequence with incompatible shape:'
            msg += f' new image shape: {image.pixels.shape} sequence shape: {self.shape}'
            raise ValueError(msg)
        if image_id in self._images_id:
            msg = 'Trying to add an image into images sequence with an already existing identifier:'
            msg += f' {image_id}'
            raise ValueError(msg)
        self.append(image)
        self._images_id.append(image_id)

    def extract_window(self, window: ImageWindowType) -> 'OrthoSequence':
        """ Extract a new images sequence by taking pixels from a window contained within the
        sequence shape.

        :param window: a window defined within the shape of this images sequence:
                       (line_start, line_stop, column_start, column_stop)
        :returns: an images sequence built with the excerpts extracted from the images over the
                  window. It has the same resolution and number of images as this sequence
                  and the image identifiers are copied from the image identifiers of this sequence.
        """
        ortho_sequence = OrthoSequence(self._delta_time_provider)
        for index, sub_tile_image in enumerate(self):
            window_image = sub_tile_image.extract_sub_image(window)
            ortho_sequence.append_image(window_image, self._images_id[index])
        return ortho_sequence
