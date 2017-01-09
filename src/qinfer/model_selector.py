#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# model_selector.py: Bayesian Model Selection module
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

from __future__ import absolute_import
from __future__ import division, unicode_literals

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ModelSelector'
]

## IMPORTS ####################################################################

from builtins import map, zip

import warnings

import numpy as np


from qinfer.smc import SMCUpdater
from qinfer.distributions import Distribution,DiscreteDistribution,IntegerValuedDistribution

## LOGGING ####################################################################

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


## CLASSES #####################################################################
class ModelSelector(Distribution):
    r"""
    Creates a new Model Selector Updater, which applies bayesian model selection to a set 
    of :class:`qinfer.smc.SMCUpdater` objects. 

    :param list updaters: list of updaters of the form [updater1,updater2,...] or 
                        [(updater_name1,updater1),(updater_name1,updater1),...] where the updater is an object of 
                        class :class:`qinfer.smc.SMCUpdater`. If no name is provided the name will become the updater's id. 
                        The models for each updater must all have the same outcome domain, and the same expparams_dtype. 
    :param qinfer.distributions.IntegerValuedDistribution prior: Prior over the models for the given updaters. 
                                                    Must be of class :class:`qinfer.distributions.IntegerValuedDistribution`, 
                                                    and have dim(len(updaters)).
    :param int n_outcome_samples: Number of outcome samples to use when estimating the future optimal experimental for model 
                                  discernment. 
    
    """
    def __init__(self,updaters, prior, n_outcome_samples=1000):

        self._updaters = []
        
        for up in updaters:
            
            if len(up)>1:
                updater_name = up[0]
                updater = up[1]
            else:
                updater = up[1]
                updater_name = id(updater)

            # check if updater is in fact an updater
            assert isinstance(SMCUpdater,updater)

            self._updaters.append((updater_name,updater))


        assert isinstance(prior,IntegerValuedDistribution)

        self._prior = prior
        self._n_outcome_samples = n_outcome_samples


    ## PROPERTIES #############################################################

    @property
    def n_outcome_samples(self):
        """
        Returns the number of outcome samples to be used when evaluating optimal 
        future experiments 

        :type: `int` 
        """
        return self._n_outcome_samples
    

    @property
    def prior(self):
        """
        Returns the prior over models. 

        :type: `qinfer.distributions.IntegerValuedDistribution`
        """
        return self._prior
    


    ## METHODS ###############################################################
    
    def _verify_exparams_dtype(self,updaters):
        exp_dtype = updaters[0].model.expparams_dtype

        for updater in updaters:
            if not expparams_dtype1 == updater.model.expparams_dtype:
                raise ValueError('All updater model do not have same expparams_dtype')

    def _verify_domains(self,updaters):
        domain = updaters[0].model.expparams_dtype

        for updater in updaters:
            if not domain == updater.model.domain:
                raise ValueError('All updater model do not have same domains')