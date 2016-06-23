#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# finite_test_models.py: Simple models for testing inference engines.
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

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.

## EXPORTS ###################################################################

__all__ = [
    'PoissonModel',
    'GaussianModel',
    'ExampleMultinomialModel'
]

## IMPORTS ###################################################################

from builtins import range

import numpy as np

from .utils import binomial_pdf

from .abstract_model import FiniteModel, DifferentiableModel
    
## CLASSES ###################################################################

class SimpleInversionModel(DifferentiableModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :math:`\left|+\right\rangle` state under a Hamiltonian :math:`H = \omega \sigma_z / 2`,
    using the interactive QLE model proposed by [WGFC13a]_.

    :param float min_freq: Minimum value for :math:`\omega` to accept as valid.
        This is used for testing techniques that mitigate the effects of
        degenerate models; there is no "good" reason to ever set this other
        than zero, other than to test with an explicitly broken model.
    """
    
    ## INITIALIZER ##

    def __init__(self, min_freq=0):
        super(SimpleInversionModel, self).__init__()
        self._min_freq = min_freq

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return [r'\omega']
        
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('w_', 'float')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a FiniteModel instance.
        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams > self._min_freq, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(SimpleInversionModel, self).likelihood(
            outcomes, modelparams, expparams
        )

        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        t = expparams['t']
        dw = modelparams - expparams['w_']
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        pr0[:, :] = np.cos(t * dw / 2) ** 2
        
        # Now we concatenate over outcomes.
        return FiniteModel.pr0_to_likelihood_array(outcomes, pr0)

    def score(self, outcomes, modelparams, expparams, return_L=False):
        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]
            
        t = expparams['t']
        dw = modelparams - expparams['w_']

        outcomes = outcomes.reshape((outcomes.shape[0], 1, 1))

        arg = dw * t / 2        
        q = (
            np.power( t / np.tan(arg), outcomes) *
            np.power(-t * np.tan(arg), 1 - outcomes)
        )[np.newaxis, ...]

        assert q.ndim == 4
        
        
        if return_L:
            return q, self.likelihood(outcomes, modelparams, expparams)
        else:
            return q



    
        
