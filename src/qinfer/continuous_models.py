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
from qinfer.abstract_model import ContinuousModel
from qinfer._lib import enum # <- TODO: replace with flufl.enum!
from qinfer.ale import binom_est_error
    
## CLASSES #####################################################################


class GaussianNoiseModel(ContinuousModel):
    __metaclass__ = ABCMeta # Needed in any class that has abstract methods.
    
    def __init__(self,sigma,Q,num_sampled_points=20,num_samples_per_point=20):
        super(GaussianNoiseModel,self).__init__(num_sampled_points,
                                                num_samples_per_point)
        self.sigma = sigma
        self._Q = Q
    @abstractmethod
    def model_function(self,modelparams,expparams):
        pass
    
    @property
    def sigma(self):
        return self._sigma
    def sigma(self,sig):
        self._sigma = sig
       
    def likelihood(self,outcomes,modelparams,expparams):
        if len(outcomes.shape) == 1:
            outcomes = outcomes[np.newaxis,:]
        
        
        like =  1/(np.sqrt(2*np.pi)*self.sigma)*np.exp(-(np.transpose(outcomes)[:,
                    np.newaxis,:]\
                    -self.model_function(
                    modelparams,expparams)[np.newaxis,:,:])**2/(2*self.sigma**2))
        return like
        
    def sample(self,weights,points,expparams,
               num_sampled_points=20,num_samples_per_point=20):
        
     
        sampled_points_in = np.random.choice(np.shape(points)[0],size=num_sampled_points,
                                         p=weights)
        sampled_points = points[sampled_points_in]
        fs = np.transpose(self.model_function(sampled_points,expparams))
      
        norm_samples = fs[...,np.newaxis] + np.random.normal(0,
                                self.sigma,fs.shape+(num_samples_per_point,))
        return norm_samples
    
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
        return np.sort(self.constant_outcome_sample(weights,modelparams,
                                            expparams).reshape(expparams.shape[0],-1))
        #sampled_points_in = np.random.choice(np.shape(modelparams)[0],
        #                    size=self.num_sampled_points*self.num_samples_per_point,
        #                                 p=weights)
        #sampled_points = modelparams[sampled_points_in]
        #fs = np.transpose(self.model_function(sampled_points,expparams)).reshape(-1)
        #return sampled_points
    