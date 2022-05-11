# -*- coding: utf-8 -*-
""" Definition of the DeltaTimeProvider abstract class and ConstantDeltaTimeProvider class

:author: GIROS Alain
:created: 02/08/2021
"""
from abc import abstractmethod, ABC
import datetime
from typing import Dict, Any, List  # @NoMove

from shapely.geometry import Point

from ..waves_exceptions import WavesException

from .localized_data_provider import LocalizedDataProvider


class NoDeltaTimeValueError(WavesException):
    """ Exception raised when a DeltaTimeProvider cannot provide a delta time at some point
    """


class NoDeltaTimeProviderError(WavesException):
    """ Exception raised when using bathymetry estimator without specifying a DeltaTimeProvider
    """


class DeltaTimeProvider(ABC, LocalizedDataProvider):
    """ A DeltaTimeProvider is a service able to provide the delta time at some position
    between two frames. The points where delta time is requested are specified by their coordinates
    in the image SRS.
    """

    @abstractmethod
    def get_delta_time(self, first_frame_id: Any, second_frame_id: Any, point: Point) -> float:
        """ Provides the delta time at some point expressed by its X and Y coordinates in some SRS,
        between 2 frames specified by their ids. The frame id definition is left undefined and
        must be specified by subclasses.

        :param first_frame_id: the id of the frame from which the duration will be counted
        :param second_frame_id: the id of the frame to which the duration will be counted
        :param point: a point expressed in the SRS coordinates set for this provider
        :returns: the delta time between frames at this point (s).
        """


# Type allowing to describe the acquisition date and time of the different frames submitted to
# the ConstantDeltaTimeProvider.
FramesTimesDict = Dict[Any, datetime.datetime]


class ConstantDeltaTimeProvider(DeltaTimeProvider):
    """ A DeltaTimeProvider which provides a constant delta time between any 2 frames whatever
    the requested position.
    """

    def __init__(self, frames_times: FramesTimesDict) -> None:
        """ Constructor

        :param frames_times: a dictionary providing the acquisition date and time of the different
                             frames. Dictionary keys are any kind of frame identifier which will be
                             used for identifying those frames when calling get_delta_time().
        """
        super().__init__()
        self._frames_times = frames_times

    def get_delta_time(self, first_frame_id: Any, second_frame_id: Any, point: Point) -> float:
        _ = point
        delta_time = self._frames_times[second_frame_id] - self._frames_times[first_frame_id]
        return delta_time.total_seconds()

    @property
    def chronology(self) -> List[Any]:
        """ :returns: a chronologically ordered list of the frames times keys.
        """
        return sorted(list(self._frames_times.keys()))
