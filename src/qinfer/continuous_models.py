#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# continuous_models.py: Models that have continuous outcomes and must therefore discretize them in some manner 
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com)
#     
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

## FEATURES ###################################################################

from __future__ import division # Ensures that a/b is always a float.

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
	'GaussianNoiseModel'
]

## IMPORTS ####################################################################

import numpy as np
from scipy.stats import binom

from qinfer.utils import binomial_pdf
from qinfer.abstract_model import ContinuousModel
from qinfer._lib import enum # <- TODO: replace with flufl.enum!
from qinfer.ale import binom_est_error
    
## CLASSES #####################################################################


class GaussianNoiseModel():
	pass