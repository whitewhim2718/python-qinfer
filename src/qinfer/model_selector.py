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

try:
    import matplotlib.pyplot as plt
except ImportError:
    import warnings
    warnings.warn("Could not import pyplot. Plotting methods will not work.")
    plt = None

try:
    import mpltools.special as mpls
except:
    # Don't even warn in this case.
    mpls = None

## LOGGING ####################################################################

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


## CLASSES #####################################################################
class ModelSelector(Distribution):
    r"""
    Creates a new Model Selector Updater, which applies bayesian model selection to a set 
    of :class:`qinfer.smc.SMCUpdater` objects. 

    :param list updaters: list of updaters of the form [updater0,updater1,...] or 
                        [(updater_name0,updater0),(updater_name1,updater1),...] where the updater is an object of 
                        class :class:`qinfer.smc.SMCUpdater`. If no name is provided the name will become the updater's id. 
                        The models for each updater must all have the same outcome domain, and the same expparams_dtype. 
    :param qinfer.distributions.IntegerValuedDistribution prior: Prior over the models for the given updaters. 
                                                    Must be of class :class:`qinfer.distributions.IntegerValuedDistribution`, 
                                                    and have dim(len(updaters)). # of distribution values must be the same as 
                                                    # of updaters. The ith value in the distribution will correspond
                                                    to the ith updater in the list of updaters. 
    :param int n_outcome_samples: Number of outcome samples to use when estimating the future optimal experimental for model 
                                  discernment. 
    
    """
    def __init__(self,updaters, prior, n_outcome_samples=1000):

        self._updaters = []
        self._n_updater_modelparams = []

        for up in updaters:
            
            if len(up)>1:
                updater_name = up[0]
                updater = up[1]
            else:
                updater = up[1]
                updater_name = id(updater)

            # check if updater is in fact an updater
            if not isinstance(SMCUpdater,updater):
                raise TypeError('''An updater provided is not actually an object of type
                                    SMCUpdater''')

            self._updaters.append((updater_name,updater))
            self._n_updater_modelparams.append(updater.model.n_modelparams)

        self._n_updater_modelparams = np.array(self._n_updater_modelparams)
        # verify domains are the same 
        self._verify_domains(self.updaters)
        self._domain = self._updaters[0][1].model.domain
        # verify expparams_dtype are the same
        self._verify_exparams_dtype(self.updaters)
        self._expparams_dtype = self._updaters[0][1].model.expparams_dtype

        if not isinstance(prior,IntegerValuedDistribution):
            raise TypeError('''The prior distribution must be an object of 
                                class IntegerValuedDistribution''')

        if not len(self.updaters)==prior:
            raise ValueError('''The number of values in the prior distribution must equal the number
                        of updaters provided.''')
        self._prior = prior
        self._model_distribution = prior
        self._n_outcome_samples = n_outcome_samples

        self._outcomes = []
        self._expparams = []


    ## PROPERTIES #############################################################
    @property
    def n_updaters(self):
        """
        Returns the number of updaters to evaluate models over. 

        :type: `int` 
        """
        return len(self.updaters)

    @property
    def n_updater_modelparams(self):
        """
        Returns the number of modelparams for the underlying model of each updater. 

        :type: `np.array` 
        """
        return self._n_updater_modelparams
    
    @property
    def n_models(self):
        """
        Returns the number of model to evaluate models over. Is the same 
        as the number of models.  

        :type: `int` 
        """

        return self.n_updaters
    
    
    @property
    def n_outcome_samples(self):
        """
        Returns the number of outcome samples to be used when evaluating optimal 
        future experiments 

        :type: `int` 
        """
        return self._n_outcome_samples
    
    @property
    def expparams_dtype(self):
        """
        Returns the exparams data type of the underlying updater models. 
        Is the same for all models.

        :type:`np.dtype`
        """
        return self._expparams_dtype
    
    @property
    def prior(self):
        """
        Returns the initial prior over models. 

        :type: `qinfer.distributions.IntegerValuedDistribution`
        """
        return self._prior

    @property
    def model_distribution(self):
        """
        Returns the current posterior probability distribution over models. 

        :type: `qinfer.distributions.IntegerValuedDistribution`
        """
        return self._model_distribution

    @property
    def bic(self):
        """
        Returns the Bayesian Information Criterion (BIC) :math:`bic(i)=ln(n)k-2ln(pr(m_i))`for the current `model_distribution`
        for the current updaters. THIS IS CURRENTLY WRONG NEED TO COME UP WITH RIGHT DERIVATION. 
        """
        probabilities = self.model_distribution.probabilities

        if len(self.expparams)>0:
            modelparams_cost = np.log(len(self.expparams))*self.n_updater_modelparams
        else:
            modelparams_cost = 0. 



        return  modelparams_cost-2*np.log(probabilities)
    
    
    @property
    def updaters(self):
        """
        Returns the list of :class:`qinfer.smc.SMCUpdater`

        :type: `list` of form [(updater_name0,updater0),(updater_name1,updater1),...]
        """
        return self._updaters

    @property
    def outcomes(self):
        """
        Returns the list of previous outcomes

        :type: `list` of [outcome0,outcome1,...]
        """
        return np.array(self._outcomes)

    @property
    def expparams(self):
        """
        Returns the list of previous expparams

        :type: `list` of [expparam0,expparam1,...]
        """
        return np.array(self._expparams,dtype=self.expparams_dtype)

    @property
    def outcome_likelihoods(self):
        """
        Returns the array of likelihoods associated with the ith (outcome,expparam) pair, 
        and the jth model of the form (n_outcomes,n_updaters).

        :type: `np.array` of form (n_outcomes,n_updaters)
        """
        return np.array(self._outcome_likelihoods)
    
    

    ## METHODS ###############################################################
    
    def _verify_exparams_dtype(self,updaters):
        exp_dtype = updaters[0][1].model.expparams_dtype

        for updater in updaters:
            if not expparams_dtype1 == updater[1].model.expparams_dtype:
                raise ValueError('All updater model do not have same expparams_dtype')

    
    def _verify_domains(self,updaters):
        domain = updaters[0][1].model.expparams_dtype

        for updater in updaters:
            if not domain == updater[1].model.domain:
                raise ValueError('All updater model do not have same domains')



    def update(self,outcome,expparams,check_for_resample=True):
        """
        Given an experiment and an outcome of that experiment, updates the
        posterior distribution of the model selectors underlying `SMCUpdaters` 
        to reflect knowledge of that experiment, and then updates the model 
        selectors prior to the posterior distribution, in order to evaluate the 
        new model knowledge. 

        After updating, resamples the underlying updaters posterior distributions
        if necessary.

        :param int outcome: Label for the outcome that was observed, as defined
            by the set of :class:`~qinfer.abstract_model.Model` instances under study.
        :param expparams: Parameters describing the experiment that was
            performed. 
        :type expparams: :class:`~numpy.ndarray` of dtype given by the
            :attr:`~qinfer.abstract_model.Model.expparams_dtype` property
            of the underlying models. Must be valid for all underyling models. 
        :param bool check_for_resample: If :obj:`True`, after performing the
            update, the effective sample size condition will be checked and
            a resampling step may be performed for underlying updaters.
        """
        self._outcomes.append(outcome)
        self._expparams.append(expparams)

        this_outcomes_likelihoods = []
        for updater in self.updaters:
            updater.update(outcome,expparams,check_for_resample=check_for_resample)

            #grab total likelihood of data under all modelparams 
            #:math:`pr(o|m)=\sum \limits_{i=1}^N pr(o|x_i,m)*\pi(x_i)`
            this_outcomes_likelihoods.append(updater.normalization_record[-1])

        this_outcomes_likelihoods = np.array(this_outcomes_likelihoods)
        self._outcome_likelihoods.append(this_outcomes_likelihoods)

        prior_probs = np.copy(self.model_distribution.probabilities)
        post_probs = this_outcomes_likelihoods*prior_probs
        post_probs = post_probs/np.sum(post_probs)
        
        self._model_distribution = IntegerValuedDistribution(prior_probs,self.n_models)
        



    ## PLOTTING ######################################################################
    
    def plot model_distribution(self,plot_BIC=False,include_names=False,xlabel=None,ylabel=None,**plot_args):
        if plot_BIC:
            hist_vals = self.bic
            if ylabel is None:
                ylabel='BIC'
        else:
            hist_vals = self.model_distribution.probabilities
            if ylabel is None:
                ylabel='probability'

        bins = self.model_distribution.values

        res = plt.hist(bins,hist_vals,weights=hist_vals,**plot_args)

        if xlabel is None:
            xlabel = 'model'
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)