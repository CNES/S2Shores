# -*- coding: utf-8 -*-
""" Class handling the information describing a waves field sample..

:author: Alain Giros
:organization: CNES
:copyright: 2021 CNES. All rights reserved.
:license: see LICENSE file
:created: 10 sep 2021
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class TemporalCorrelationWavesFieldEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a bathymetry sample by a
    TemporalCorrelationBathyEstimator.
    """
