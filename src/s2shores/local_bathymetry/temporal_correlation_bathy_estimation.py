# -*- coding: utf-8 -*-
""" Class handling the information describing a wave field sample..

:authors: see AUTHORS file
:organization: CNES, LEGOS, SHOM
:copyright: 2024 CNES. All rights reserved.
:created: 10 September 2021
:license: see LICENSE file


  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
  in compliance with the License. You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License
  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
  or implied. See the License for the specific language governing permissions and
  limitations under the License.
"""
from ..data_model.bathymetry_sample_estimation import BathymetrySampleEstimation


class TemporalCorrelationBathyEstimation(BathymetrySampleEstimation):
    """ This class encapsulates the information estimated in a bathymetry sample by a
    TemporalCorrelationBathyEstimator.
    """
