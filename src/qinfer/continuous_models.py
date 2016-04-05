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
from abc import ABCMeta,abstractmethod,abstractproperty
from qinfer.utils import binomial_pdf
from qinfer.abstract_model import ContinuousModel, DifferentiableModel
from qinfer._lib import enum # <- TODO: replace with flufl.enum!
from qinfer.ale import binom_est_error
import math 
import warnings

NUMBA_AVAILABLE = False
try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    warnings.warn('''Could not import Numbda. Numba support 
        will be disabled. 
        ''')    

## CLASSES #####################################################################

with warnings.catch_warnings():
    warnings.filterwarnings("module",category=ImportWarning)

class GaussianNoiseModel(ContinuousModel):#,DifferentiableModel):
    __metaclass__ = ABCMeta # Needed in any class that has abstract methods.


    
    def __init__(self,sigma,Q,num_sampled_points=20,num_samples_per_point=20,
                    use_numba=False,parallelize='cpu'):
        super(GaussianNoiseModel,self).__init__(num_sampled_points,
                                                num_samples_per_point)
        self.sigma = sigma
        self._Q = Q
        
        self.use_numba = use_numba
        if not NUMBA_AVAILABLE:
            self.use_numba = False 

        assert parallelize in ('cpu','parallel','cuda')

        self.parallelize = parallelize




        self._outcome_stochastic_component = np.random.normal(0,self.sigma,
        										(self.num_sampled_points,num_samples_per_point)	)
        
        self._outcome_sampled_points = np.zeros((self.num_sampled_points,self.n_modelparams))


    @abstractmethod
    def model_function(self,modelparams,expparams):
        pass
    
    @abstractmethod
    def model_function_derivative(self,modelparams,expparams):
        pass

    @property
    def sigma(self):
        return self._sigma
    def sigma(self,sig):
        self._sigma = sig
       
    def likelihood(self,outcomes,modelparams,expparams):
        if len(outcomes.shape) == 1:
            outcomes = outcomes[np.newaxis,:]
        
        
        if self.use_numba:
            return GaussianNoiseModel._numba_likelihood(self.model_function,self.sigma,
                    outcomes,modelparams,expparams)
        else:
            return GaussianNoiseModel._numpy_likelihood(self.model_function,self.sigma,
                    outcomes,modelparams,expparams)
    
    @staticmethod
    def _numpy_likelihood(model_function,sigma,outcomes,modelparams,expparams):
            return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(np.transpose(outcomes)[:,
                    np.newaxis,:]\
                    -model_function(
                    modelparams,expparams)[np.newaxis,:,:])**2/(2*sigma**2))


    if NUMBA_AVAILABLE:
        @staticmethod
        @nb.jit
        def _numba_likelihood(model_function,sigma,outcomes,modelparams,expparams):
            model_func_results = model_function(modelparams,expparams)
            tran_outcomes = np.transpose(outcomes)
            
           
            return GaussianNoiseModel._numba_nopython_likelihood_component(sigma,tran_outcomes,
                                        model_func_results)

        @staticmethod
        @nb.guvectorize([(nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:,:])],
                        '(),(x,z),(y,z)->(x,y,z)',nopython=True,target='cpu')
        def _numba_nopython_likelihood_component(sigma,tran_outcomes,model_func_results,res):
            mul_const = 1/(math.sqrt(2*math.pi)*sigma[0])
            scal_const = 2*sigma[0]**2
            for x in range(tran_outcomes.shape[0]):
                for y in range(model_func_results.shape[0]):
                    for z in range(model_func_results.shape[1]):
                        res[x,y,z] = mul_const*math.exp(-(tran_outcomes[x,z]-\
                            model_func_results[y,z])**2/scal_const)
    else:
        #numba is not available revert 
        @staticmethod
        def _numba_likelihood(model_function,sigma,outcomes,modelparams,expparams):
            warnings.warn('Numbda is not available, reverting to numpy implementation')
            return _numpy_likelihood(model_function,sigma,outcomes,modelparams,expparams)


    def sample(self,weights,modelparams,expparams,
               num_sampled_points=20,num_samples_per_point=20):
        
     
        sampled_points_in = np.random.choice(np.shape(modelparams)[0],size=num_sampled_points,
                                         p=weights)
        sampled_points = points[sampled_points_in]
        fs = np.transpose(self.model_function(sampled_points,expparams))
      
        norm_samples = fs[...,np.newaxis] + np.random.normal(0,
                                self.sigma,fs.shape+(num_samples_per_point,))
        return norm_samples
    
    def score(self,outcomes,modelparams,expparams,return_L=False):
    	r"""
        Returns the score of this likelihood function, defined as:
        
        .. math::
        
            q(d, \vec{x}; \vec{e}) = \vec{\nabla}_{\vec{x}} \log \Pr(d | \vec{x}; \vec{e}).
            
        Calls are represented as a four-index tensor
        ``score[idx_modelparam, idx_outcome, idx_model, idx_experiment]``.
        The left-most index may be suppressed for single-parameter models.
        
        If return_L is True, both `q` and the likelihood `L` are returned as `q, L`.
        """
    	fs = self.model_function(outcomes,expparams)
    	
    	np.log()

    	if return_L:
    		return q, self.likelihood(outcomes,modelparams,expparams)
    	else:
    		return q 
    @property
    def is_n_outcomes_constant(self):
        return False
    
    def constant_outcome_sample(self,weights,modelparams,expparams):

        return self.sample(weights,modelparams,expparams,num_sampled_points=self.num_sampled_points,
                num_samples_per_point=self.num_samples_per_point)
    
    def simulate_experiment(self,modelparams,expparams,repeat=1):
        fs = self.model_function(sampled_points,expparams)
        return fs + np.random.normal(0,self.sigma,fs.shape)
    
    
    def outcomes(self,weights,modelparams,expparams):
        #return np.sort(self.constant_outcome_sample(weights,modelparams,
        #                                    expparams).reshape(expparams.shape[0],-1))
        #sampled_points_in = np.random.choice(np.shape(modelparams)[0],
        #                    size=self.num_sampled_points*self.num_samples_per_point,
        #                                 p=weights)
        #sampled_points = modelparams[sampled_points_in]
        #fs = np.transpose(self.model_function(sampled_points,expparams)).reshape(-1)
        #return sampled_points
    	
    	fs = np.transpose(self.model_function(self._outcome_sampled_points,expparams))
    	norm_samples = fs[...,np.newaxis]+self._outcome_stochastic_component
    	return np.sort(norm_samples.reshape(expparams.shape[0],-1))
    
    def update_callback(self, weights, modelparams, expparams= None):
        """
        Callback function that will be called by the SMC updater 
        at every update. By default does nothing. 

        :param np.ndarray weights: Set of weights with a weight
            corresponding to every modelparam. 
        :param np.ndarray modelparams: Set of model parameter vectors to be
            updated.
        :param np.ndarray expparams: An experiment parameter array describing
            the experiment that was just performed.

        """
    	self._outcome_stochastic_component = np.random.normal(0,self.sigma,
    										(self.num_sampled_points,self.num_samples_per_point))
    	sampled_indexes = np.random.choice(np.shape(modelparams)[0],size=self.num_sampled_points,
                                     p=weights)
    	self._outcome_sampled_points = modelparams[sampled_indexes]