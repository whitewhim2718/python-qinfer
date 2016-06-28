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
    'MultinomialModel'
]

## IMPORTS ###################################################################

from builtins import range

import numpy as np
from scipy.special import gammaln
from .utils import binomial_pdf

from .abstract_model import Model, DifferentiableModel
    
## CLASSES ###################################################################

class PoissonModel(Model):
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

    def __init__(self, num_outcome_samples=500):
        super(PoissonModel, self).__init__()
        self.num_outcome_samples = num_outcome_samples

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return [r'\lambda']
        
    @property
    def expparams_dtype(self):
        return []
    
    @property
    def outcomes_dtype(self):
        return int
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return self.num_outcome_samples
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.

        super(PoissonModel, self).likelihood(outcomes, modelparams, expparams)

       
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
        
        if len(outcomes.shape) == 1:
            outcomes = outcomes[..., np.newaxis]

        lamb_da = modelparams[np.newaxis,...]
        outcomes = outcomes[:,np.newaxis,:]
        return np.exp(outcomes*np.log(lamb_da)-gammaln(outcomes+1)-lamb_da)


    def score(self, outcomes, modelparams, expparams, return_L=False):
        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]
        
        return super(PoissonModel, self).score(outcomes, modelparams, expparams, return_L) 
        t = expparams['t']
        dw = modelparams - expparams['w_']

        outcomes_reshaped = outcomes[np.newaxis,:,np.newaxis,np.newaxis]
        modelparams_reshaped = modelparams[:,np.newaxix,:,:]
        scr = (outcomes*np.pow(modelparams,outcomes-1)*np.exp(-modelparams)-\
                modelparams*np.pow(modelparams,outcomes))/factorial(outcomes)
        
        scr = outcomes/lamb_da-1
        
        if return_L:
            return scr, self.likelihood(outcomes, modelparams, expparams)
        else:
            return scr


    def simulate_experiment(self,modelparams,expparams,repeat=1):

        super(PoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]    
        
        modelparams = modelparams.reshape(-1)
        outcomes = np.random.poisson(modelparams,(expparams.shape[0],modelparams.shape[0]))

        return outcomes 

class GaussianModel(DifferentiableModel):
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

    def __init__(self, num_outcome_samples=400):
        super(PoissonModel, self).__init__()
        self.num_outcome_samples = num_outcome_samples

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return [r'\lambda']
        
    @property
    def expparams_dtype(self):
        return []
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return self.num_outcome_samples
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.

        super(PoissonModel, self).likelihood(outcomes, modelparams, expparams)

       
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]

        lamb_da = modelparams[np.newaxis,...]
        outcomes = outcomes[:,np.newaxis,:]
        return lamb_da**(outcomes)*np.exp(-lamb_da)/np.misc.factorial(outcomes)


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


class MultinomialModel(DifferentiableModel):
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

    def __init__(self, num_outcome_samples=400):
        super(PoissonModel, self).__init__()
        self.num_outcome_samples = num_outcome_samples

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return [r'\lambda']
        
    @property
    def expparams_dtype(self):
        return []
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return self.num_outcome_samples
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.

        super(PoissonModel, self).likelihood(outcomes, modelparams, expparams)

       
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]

        lamb_da = modelparams[np.newaxis,...]
        outcomes = outcomes[:,np.newaxis,:]
        return lamb_da**(outcomes)*np.exp(-lamb_da)/np.misc.factorial(outcomes)


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



    
        
