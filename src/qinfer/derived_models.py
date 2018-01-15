#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# derived_models.py: Models that decorate and extend other models.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division # Ensures that a/b is always a float.

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'DerivedModel',
    'PoisonedModel',
    'ReferencedPoissonModel',
    'BinomialModel',
    'MultinomialModel',
    'MLEModel',
    'RandomWalkModel',
    'GaussianRandomWalkModel'
]

## IMPORTS ####################################################################

from builtins import range
from functools import reduce
from past.builtins import basestring

import numpy as np
from scipy.stats import binom, multivariate_normal
from scipy.special import xlogy, gammaln
from itertools import combinations_with_replacement as tri_comb

from qinfer.utils import binomial_pdf, multinomial_pdf, sample_multinomial
from qinfer.abstract_model import Model, FiniteOutcomeModel, DifferentiableModel
from qinfer._lib import enum # <- TODO: replace with flufl.enum!
from qinfer.utils import binom_est_error
from qinfer.domains import IntegerDomain, MultinomialDomain

## FUNCTIONS ###################################################################

def rmfield( a, *fieldnames_to_remove ):
    # Removes named fields from a structured np array
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]

def poisson_pdf(k, mu):
    """
    Probability of k events in a poisson process of expectation mu
    """
    return np.exp(xlogy(k, mu) - gammaln(k + 1) - mu)

## CLASSES #####################################################################

class DerivedModel(Model):
    """
    Base class for any model that decorates another model.
    Provides passthroughs for modelparam_names, n_modelparams, etc.

    Many of these passthroughs can and should be overriden by
    specific subclasses, but it is rare that something will
    override all of them.
    """
    _underlying_model = None
    def __init__(self, underlying_model):
        self._underlying_model = underlying_model
        super(DerivedModel, self).__init__()
    
    @property
    def underlying_model(self):
        return self._underlying_model

    @property
    def base_model(self):
        return self._underlying_model.base_model

    @property
    def model_chain(self):
        return self._underlying_model.model_chain + (self._underlying_model, )

    @property
    def n_modelparams(self):
        # We have as many modelparameters as the underlying model.
        return self.underlying_model.n_modelparams

    @property
    def expparams_dtype(self):
        return self.underlying_model.expparams_dtype
        
    @property
    def modelparam_names(self):
        return self.underlying_model.modelparam_names

    @property
    def Q(self):
        return self.underlying_model.Q
    
    def clear_cache(self):
        self.underlying_model.clear_cache()

    def n_outcomes(self, expparams):
        return self.underlying_model.n_outcomes(expparams)
    
    def are_models_valid(self, modelparams):
        return self.underlying_model.are_models_valid(modelparams)

    def domain(self, expparams):
        return self.underlying_model.domain(expparams)
    
    def are_expparam_dtypes_consistent(self, expparams):
        return self.underlying_model.are_expparam_dtypes_consistent(expparams)
    
    def update_timestep(self, modelparams, expparams):
        return self.underlying_model.update_timestep(modelparams, expparams)

    def canonicalize(self, modelparams):
        return self.underlying_model.canonicalize(modelparams)

    def update_callback(self,weights,modelparams):
        return self.underlying_model.update_callback(weights,modelparams)
    
    def representative_outcomes(self, weights, modelparams, expparams,likelihood_modelparams=None,
                                likelihood_weights=None):
        return self.underlying_model.representative_outcomes(weights, modelparams, expparams,likelihood_modelparams,
                                likelihood_weights)
        
    def simulate_experiment(self,modelparams,expparams,repeat=1):
        # It might be tempting to call this for sim_count incrementing, 
        # but it will almost certainly be a waste of time. It might even 
        # be called in the concrete implementation as a part of simulating 
        # the derived model. Moreover, it might cause an error if,
        # for example, the number of model parameters changes.
        #return self.underlying_model.simulate_experiment(modelparams,expparams,repeat)
        self._sim_count += modelparams.shape[0] * expparams.shape[0] * repeat

PoisonModes = enum.enum("ALE", "MLE")

class PoisonedModel(DerivedModel, FiniteOutcomeModel):
    r"""
    FiniteOutcomeModel that simulates sampling error incurred by the MLE or ALE methods of
    reconstructing likelihoods from sample data. The true likelihood given by an
    underlying model is perturbed by a normally distributed random variable
    :math:`\epsilon`, and then truncated to the interval :math:`[0, 1]`.
    
    The variance of :math:`\epsilon` can be specified either as a constant,
    to simulate ALE (in which samples are collected until a given threshold is
    met), or as proportional to the variance of a possibly-hedged binomial
    estimator, to simulate MLE.
    
    :param Model underlying_model: The "true" model to be poisoned.
    :param float tol: For ALE, specifies the given error tolerance to simulate.
    :param int n_samples: For MLE, specifies the number of samples collected.
    :param float hedge: For MLE, specifies the hedging used in estimating the
        true likelihood.
    """
    def __init__(self, underlying_model,
        tol=None, n_samples=None, hedge=None
    ):
        super(PoisonedModel, self).__init__(underlying_model)
        
        if tol is None != n_samples is None:
            raise ValueError(
                "Exactly one of tol and n_samples must be specified"
            )
        
        if tol is not None:
            self._mode = PoisonModes.ALE
            self._tol = tol
        else:
            self._mode = PoisonModes.MLE
            self._n_samples = n_samples
            self._hedge = hedge if hedge is not None else 0.0
            
    ## METHODS ##

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        
        # Get the original, undisturbed likelihoods.
        super(PoisonedModel, self).likelihood(outcomes, modelparams, expparams)
        L = self.underlying_model.likelihood(
            outcomes, modelparams, expparams)
            
        # Now get the random variates from a standard normal [N(0, 1)]
        # distribution; we'll rescale them soon.
        epsilon = np.random.normal(size=L.shape)
        
        # If ALE, rescale by a constant tolerance.
        if self._mode == PoisonModes.ALE:
            epsilon *= self._tol
        # Otherwise, rescale by the estimated error in the binomial estimator.
        elif self._mode == PoisonModes.MLE:
            epsilon *= binom_est_error(p=L, N=self._n_samples, hedge=self._hedge)
        
        # Now we truncate and return.
        np.clip(L + epsilon, 0, 1, out=L)
        return L
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        """
        Simulates experimental data according to the original (unpoisoned)
        model. Note that this explicitly causes the simulated data and the
        likelihood function to disagree. This is, strictly speaking, a violation
        of the assumptions made about `~qinfer.abstract_model.Model` subclasses.
        This violation is by intention, and allows for testing the robustness
        of inference algorithms against errors in that assumption.
        """
        super(PoisonedModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams, expparams, repeat)

class ReferencedPoissonModel(DerivedModel):
    """
    Model whose "true" underlying model is a coin flip, but where the coin is
    only accessible by drawing three poisson random variates, the rate
    of the third being the convex combination of the rates of the first two,
    and the linear parameter being the weight of the coin.
    By drawing from all three poisson random variables, information about the
    coin can be extracted, and the rates are thought of as nuisance parameters.
    This model is in many ways similar to the :class:`BinomialModel`, in
    particular, it requires the underlying model to have exactly two outcomes.
    :param Model underlying_model: The "true" model hidden by poisson random
        variables which set upper and lower references.
    Note that new ``modelparam`` fields alpha and beta are
    added by this model. They refer to, respectively, the higher poisson
    rate (corresponding to underlying probability 1)
    and the lower poisson rate (corresponding to underlying probability 0).
    Additionally, an exparam field ``mode`` is added.
    This field indicates whether just the signal has been measured (0), the
    bright reference (1), or the dark reference (2).
    To ensure the correct operation of this model, it is important that the
    decorated model does not also admit a field with the name ``mode``.
    """

    SIGNAL = 0
    BRIGHT = 1
    DARK = 2

    def __init__(self, underlying_model):
        super(ReferencedPoissonModel, self).__init__(underlying_model)

        if not (underlying_model.is_outcomes_constant and underlying_model.domain(None).n_members == 2):
            raise ValueError("Decorated model must be a two-outcome model.")

        if isinstance(underlying_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "p".
            self._expparams_scalar = True
            self._expparams_dtype = [('p', underlying_model.expparams_dtype), ('mode', 'int')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = underlying_model.expparams_dtype + [('mode', 'int')]

        # The domain for any mode of an experiment is all of the non-negative integers
        self._domain = IntegerDomain(min=0)

    ## PROPERTIES ##

    @property
    def n_modelparams(self):
        return super(ReferencedPoissonModel, self).n_modelparams + 2

    @property
    def modelparam_names(self):
        underlying_names = super(ReferencedPoissonModel, self).modelparam_names
        return underlying_names + [r'\alpha', r'\beta']

    @property
    def expparams_dtype(self):
        return self._expparams_dtype

    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True

    ## METHODS ##

    def are_models_valid(self, modelparams):
        u_valid = self.underlying_model.are_models_valid(modelparams[:,:-2])
        s_valid = np.logical_and(modelparams[:,-1] <= modelparams[:,-2], modelparams[:,-2] >= 0)
        return np.logical_and(u_valid, s_valid)

    def canonicalize(self, modelparams):
        u_model = self.underlying_model.canonicalize(modelparams[:,:-2])
        mask = modelparams[:,-2] <= modelparams[:,-1]
        avg = (modelparams[mask,-1] + modelparams[mask,-2]) / 2 
        modelparams[mask,-2] = avg
        modelparams[mask,-1] = avg
        return modelparams

    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        Note: This is incorrect as there are an infinite number of outcomes.
        We arbitrarily pick a number.
        """
        return 1000

    def domain(self, expparams):
        """
        Returns a list of ``Domain``s, one for each input expparam.
        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        :rtype: list of ``Domain``
        """
        return self._domain if expparams is None else [self._domain for ep in expparams]

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(ReferencedPoissonModel, self).likelihood(outcomes, modelparams, expparams)

        L = np.empty((outcomes.shape[0], modelparams.shape[0], expparams.shape[0]))
        ot = np.tile(outcomes, (modelparams.shape[0],1)).T

        for idx_ep, expparam in enumerate(expparams):

            if expparam['mode'] == self.SIGNAL:

                # Get the probability of outcome 1 for the underlying model.
                pr0 = self.underlying_model.likelihood(
                    np.array([0], dtype='uint'),
                    modelparams[:,:-2],
                    np.array([expparam['p']]) if self._expparams_scalar else np.array([expparam]))[0,:,0]
                pr0 = np.tile(pr0, (outcomes.shape[0], 1))

                # Reference Rate
                alpha = np.tile(modelparams[:, -2], (outcomes.shape[0], 1))
                beta = np.tile(modelparams[:, -1], (outcomes.shape[0], 1))

                # For each model parameter, turn this into an expected poisson rate
                gamma = pr0 * alpha + (1 - pr0) * beta

                # The likelihood of getting each of the outcomes for each of the modelparams
                L[:,:,idx_ep] = poisson_pdf(ot, gamma)

            elif expparam['mode'] == self.BRIGHT:

                # Reference Rate
                alpha = np.tile(modelparams[:, -2], (outcomes.shape[0], 1))

                # The likelihood of getting each of the outcomes for each of the modelparams
                L[:,:,idx_ep] = poisson_pdf(ot, alpha)

            elif expparam['mode'] == self.DARK:

                # Reference Rate
                beta = np.tile(modelparams[:, -1], (outcomes.shape[0], 1))

                # The likelihood of getting each of the outcomes for each of the modelparams
                L[:,:,idx_ep] = poisson_pdf(ot, beta)
            else:
                raise(ValueError('Unknown mode detected in ReferencedPoissonModel.'))

        assert not np.any(np.isnan(L))
        return L

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(ReferencedPoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]
        outcomes = np.empty(shape=(repeat, n_mps, n_eps))

        for idx_ep, expparam in enumerate(expparams):
            if expparam['mode'] == self.SIGNAL:
                # Get the probability of outcome 1 for the underlying model.
                print(self._expparams_scalar)
 
                ep = np.array([expparam['p']]) if self._expparams_scalar else np.array([expparam])
                print(ep.shape)
                pr0 = self.underlying_model.likelihood(
                    np.array([0], dtype='uint'),
                    modelparams[:,:-2],
                    ep)[0,:,0]

                # Reference Rate
                alpha = modelparams[:, -2]
                beta = modelparams[:, -1]

                outcomes[:,:,idx_ep] = np.random.poisson(pr0 * alpha + (1 - pr0) * beta, size=(repeat, n_mps))
            elif expparam['mode'] == self.BRIGHT:
                alpha = modelparams[:, -2]
                outcomes[:,:,idx_ep] = np.random.poisson(alpha, size=(repeat, n_mps))
            elif expparam['mode'] == self.DARK:
                beta = modelparams[:, -1]
                outcomes[:,:,idx_ep] = np.random.poisson(beta, size=(repeat, n_mps))
            else:
                raise(ValueError('Unknown mode detected in ReferencedPoissonModel.'))

        return outcomes[0,0,0] if outcomes.size == 1 else outcomes

    def update_timestep(self, modelparams, expparams):
        return self.underlying_model.update_timestep(modelparams,
            np.array([expparams['p']]) if self._expparams_scalar else expparams
        )


class BinomialModel(DerivedModel,FiniteOutcomeModel):
    """
    Model representing finite numbers of iid samples from another model,
    using the binomial distribution to calculate the new likelihood function.
    
    :param qinfer.abstract_model.Model underlying_model: An instance of a two-
        outcome model to be decorated by the binomial distribution.
        
    Note that a new experimental parameter field ``n_meas`` is added by this
    model. This parameter field represents how many times a measurement should
    be made at a given set of experimental parameters. To ensure the correct
    operation of this model, it is important that the decorated model does not
    also admit a field with the name ``n_meas``.
    """
    
    def __init__(self, underlying_model):
        super(BinomialModel, self).__init__(underlying_model)
        
        if not (underlying_model.is_n_outcomes_constant and underlying_model.n_outcomes(None) == 2):
            raise ValueError("Decorated model must be a two-outcome model.")
        
        if isinstance(underlying_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "x".
            self._expparams_scalar = True
            self._expparams_dtype = [('x', underlying_model.expparams_dtype), ('n_meas', 'uint')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = underlying_model.expparams_dtype + [('n_meas', 'uint')]
    
    ## PROPERTIES ##
        
    @property
    def decorated_model(self):
        # Provided for backcompat only.
        return self.underlying_model
    

    @property
    def expparams_dtype(self):
        return self._expparams_dtype
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return False
    
    ## METHODS ##
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return expparams['n_meas'] + 1

    def domain(self, expparams):
        """
        Returns a list of ``Domain``s, one for each input expparam.

        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property, or, in the case where ``n_outcomes_constant`` is ``True``,
            ``None`` should be a valid input.

        :rtype: list of ``Domain``
        """
        return [IntegerDomain(min=0,max=n_o-1) for n_o in self.n_outcomes(expparams)]
    
    def are_expparam_dtypes_consistent(self, expparams):
        """
        Returns `True` iff all of the given expparams 
        correspond to outcome domains with the same dtype.
        For efficiency, concrete subclasses should override this method 
        if the result is always `True`.

        :param np.ndarray expparams: Array of expparamms 
             of type `expparams_dtype`
        :rtype: `bool`
        """
        # The output type is always the same, even though the domain is not.
        return True

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(BinomialModel, self).likelihood(outcomes, modelparams, expparams)
        pr1 = self.underlying_model.likelihood(
            np.array([1], dtype='uint'),
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams)
        
        # Now we concatenate over outcomes.
        L = np.concatenate([
            binomial_pdf(expparams['n_meas'][np.newaxis, :], outcomes[idx], pr1)
            for idx in range(outcomes.shape[0])
            ]) 
        assert not np.any(np.isnan(L))
        return L
            
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        # FIXME: uncommenting causes a slowdown, but we need to call
        #        to track sim counts.
        #super(BinomialModel, self).simulate_experiment(modelparams, expparams)
        
        # Start by getting the pr(1) for the underlying model.
        pr1 = self.underlying_model.likelihood(
            np.array([1], dtype='uint'),
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams)
            
        dist = binom(
            expparams['n_meas'].astype('int'), # ← Really, NumPy?
            pr1[0, :, :]
        )
        sample = (
            (lambda: dist.rvs()[np.newaxis, :, :])
            if pr1.size != 1 else
            (lambda: np.array([[[dist.rvs()]]]))
        )
        os = np.concatenate([
            sample()
            for idx in range(repeat)
        ], axis=0)
        return os[0,0,0] if os.size == 1 else os
        
    def update_timestep(self, modelparams, expparams):
        return self.underlying_model.update_timestep(modelparams,
            expparams['x'] if self._expparams_scalar else expparams
        )

    def score(self, outcomes, modelparams, expparams, return_L=False):
        super(BinomialModel, self).score(outcomes, modelparams, expparams, return_L) 

        pr1 = self.underlying_model.likelihood(
            np.array([1], dtype='uint'),
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams)[np.newaxis,:,:,:]

        scr_underlying = self.underlying_model.score(np.array([1], dtype='uint'),modelparams,
                            expparams['x'] if self._expparams_scalar else expparams)

        
class DifferentiableBinomialModel(BinomialModel, DifferentiableModel):
    """
    Extends :class:`BinomialModel` to take advantage of differentiable
    two-outcome models.
    """
    
    def __init__(self, underlying_model):
        if not isinstance(underlying_model, DifferentiableModel):
            raise TypeError("Decorated model must also be differentiable.")
        BinomialModel.__init__(self, underlying_model)
    
    def score(self, outcomes, modelparams, expparams):
        raise NotImplementedError("Not yet implemented.")
        
    def fisher_information(self, modelparams, expparams):
        # Since the FI simply adds, we can multiply the single-shot
        # FI provided by the underlying model by the number of measurements
        # that we perform.
        two_outcome_fi = self.underlying_model.fisher_information(
            modelparams, expparams
        )
        return two_outcome_fi * expparams['n_meas']

        if return_L:
            return scr, self.likelihood(outcomes,modelparams,expparams)
        else:
            return scr





class MultinomialModel(DerivedModel, FiniteOutcomeModel):
    """
    Model representing finite numbers of iid samples from another model,
    using the multinomial distribution to calculate the new likelihood function.
    
    :param qinfer.abstract_model.FiniteOutcomeModel underlying_model: An instance 
        of a D-outcome model to be decorated by the multinomial distribution. 
        This underlying model must have ``n_outcomes_constant`` as ``True``.
        
    Note that a new experimental parameter field ``n_meas`` is added by this
    model. This parameter field represents how many times a measurement should
    be made at a given set of experimental parameters. To ensure the correct
    operation of this model, it is important that the decorated model does not
    also admit a field with the name ``n_meas``.
    """
    
    ## INITIALIZER ##

    def __init__(self, underlying_model):
        super(MultinomialModel, self).__init__(underlying_model)

        if isinstance(underlying_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "x".
            self._expparams_scalar = True
            self._expparams_dtype = [('x', underlying_model.expparams_dtype), ('n_meas', 'uint')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = underlying_model.expparams_dtype + [('n_meas', 'uint')]

        # Demand that the underlying model always has the same number of outcomes
        # This assumption could in principle be generalized, but not worth the effort now.
        assert(self.underlying_model.is_n_outcomes_constant)
        self._underlying_domain = self.underlying_model.domain(None)
        self._n_sides = self._underlying_domain.n_members
        # Useful for getting the right type, etc.
        self._example_domain = MultinomialDomain(n_elements=self.n_sides, n_meas=3) 

    ## PROPERTIES ##
    

    @property
    def decorated_model(self):
        # Provided for backcompat only.
        return self.underlying_model

    @property
    def expparams_dtype(self):
        return self._expparams_dtype  
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        # Different values of n_meas result in different numbers of outcomes
        return False
    
    ## METHODS ##

    @property
    def n_sides(self):
        """
        Returns the number of possible outcomes of the underlying model.
        """
        return self._n_sides

    @property
    def underlying_domain(self):
        """
        Returns the `Domain` of the underlying model.
        """
        return self._underlying_domain
    
    ## METHODS ##

    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        # Standard combinatorial formula equal to the number of 
        # possible tuples whose non-negative integer entries sum to n_meas.
        n = expparams['n_meas']
        k = self.n_sides
        return scipy.special.binom(n + k - 1, k - 1)

    def domain(self, expparams):
        """
        Returns a list of :class:`Domain` objects, one for each input expparam.
        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        :rtype: list of ``Domain``
        """
        return [
            MultinomialDomain(n_elements=self.n_sides, n_meas=ep['n_meas']) 
                for ep in expparams
        ]
    
    def are_expparam_dtypes_consistent(self, expparams):
        """
        Returns `True` iff all of the given expparams 
        correspond to outcome domains with the same dtype.
        For efficiency, concrete subclasses should override this method 
        if the result is always `True`.

        :param np.ndarray expparams: Array of expparamms 
             of type `expparams_dtype`
        :rtype: `bool`
        """
        # The output type is always the same, even though the domain is not.
        return True
  
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(MultinomialModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Save a wee bit of time by only calculating the likelihoods of outcomes 0,...,d-2
        prs = self.underlying_model.likelihood(
            self.underlying_domain.values[:-1],
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams) 
            # shape (sides-1, n_mps, n_eps)
        
        prs = np.tile(prs, (outcomes.shape[0],1,1,1)).transpose((1,0,2,3))
        # shape (n_outcomes, sides-1, n_mps, n_eps)

        os = self._example_domain.to_regular_array(outcomes)
        # shape (n_outcomes, sides)
        os = np.tile(os, (modelparams.shape[0],expparams.shape[0],1,1)).transpose((3,2,0,1))
        # shape (n_outcomes, sides, n_mps, n_eps)

        L = multinomial_pdf(os, prs) 
        assert not np.any(np.isnan(L))
        return L

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(MultinomialModel, self).simulate_experiment(modelparams, expparams)
        
        n_sides = self.n_sides
        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]

        # Save a wee bit of time by only calculating the likelihoods of outcomes 0,...,d-2
        prs = np.empty((n_sides,n_mps,n_eps))
        prs[:-1,...] = self.underlying_model.likelihood(
            self.underlying_domain.values[:-1],
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams) 
            # shape (sides, n_mps, n_eps)


        os = np.concatenate([
            sample_multinomial(n_meas, prs[:,:,idx_n_meas], size=repeat)[np.newaxis,...]
            for idx_n_meas, n_meas in enumerate(expparams['n_meas'].astype('int'))
        ]).transpose((3,2,0,1))

        # convert to fancy data type
        os = self._example_domain.from_regular_array(os)

        return os[0,0,0] if os.size == 1 else os

        
class DifferentiableBinomialModel(BinomialModel, DifferentiableModel):
    """
    Extends :class:`BinomialModel` to take advantage of differentiable
    two-outcome models.
    """
    
    def __init__(self, underlying_model):
        if not isinstance(underlying_model, DifferentiableModel):
            raise TypeError("Decorated model must also be differentiable.")
        BinomialModel.__init__(self, underlying_model)
    
    def score(self, outcomes, modelparams, expparams):
        raise NotImplementedError("Not yet implemented.")
        
    def fisher_information(self, modelparams, expparams):
        # Since the FI simply adds, we can multiply the single-shot
        # FI provided by the underlying model by the number of measurements
        # that we perform.
        two_outcome_fi = self.underlying_model.fisher_information(
            modelparams, expparams
        )
        return two_outcome_fi * expparams['n_meas']

class MLEModel(DerivedModel):
    r"""
    Uses the method of [JDD08]_ to approximate the maximum likelihood
    estimator as the mean of a fictional posterior formed by amplifying the
    Bayes update by a given power :math:`\gamma`. As :math:`\gamma \to
    \infty`, this approximation to the MLE improves, but at the cost of
    numerical stability.

    :param float likelihood_power: Power to which the likelihood calls
        should be rasied in order to amplify the Bayes update.

    .. [JDD08] Particle methods for maximum likelihood estimation
        in latent variable models. :doi:`0.1007/s11222-007-9037-8`.
    """

    def __init__(self, underlying_model, likelihood_power):
        super(MLEModel, self).__init__(underlying_model)
        self._pow = likelihood_power

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(MLEModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams, expparams, repeat)

    def likelihood(self, outcomes, modelparams, expparams):
        L = self.underlying_model.likelihood(outcomes, modelparams, expparams)
        return L**self._pow

class RandomWalkModel(DerivedModel):
    r"""
    Model such that after each time step, a random perturbation is added to
    each model parameter vector according to a given distribution.
    
    :param Model underlying_model: Model representing the likelihood with no
        random walk added.
    :param Distribution step_distribution: Distribution over step vectors.
    """
    def __init__(self, underlying_model, step_distribution):
        self._step_dist = step_distribution
        
        super(RandomWalkModel, self).__init__(underlying_model)
        
        if self.underlying_model.n_modelparams != self._step_dist.n_rvs:
            raise TypeError("Step distribution does not match model dimension.")
        
            
    ## METHODS ##
    
    def likelihood(self, outcomes, modelparams, expparams):
        super(RandomWalkModel, self).likelihood(outcomes, modelparams, expparams)
        return self.underlying_model.likelihood(outcomes, modelparams, expparams)
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(RandomWalkModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams, expparams, repeat)
        
    def update_timestep(self, modelparams, expparams):
        # Note that the timestep update is presumed to be independent of the
        # experiment.
        steps = self._step_dist.sample(n=modelparams.shape[0] * expparams.shape[0])
        # Break apart the first two axes and transpose.
        steps = steps.reshape((modelparams.shape[0], expparams.shape[0], self.n_modelparams))
        steps = steps.transpose((0, 2, 1))
        
        return modelparams[:, :, np.newaxis] + steps
        
class GaussianRandomWalkModel(DerivedModel):
    r"""
    Model such that after each time step, a random perturbation is 
    added to each model parameter vector according to a 
    zero-mean gaussian distribution.
    
    The :math:`n\times n` covariance matrix of this distribution is 
    either fixed and known, or its entries are treated as unknown, 
    being appended to the model parameters.
    For diagonal covariance matrices, :math:`n` parameters are added to the model 
    storing the square roots of the diagonal entries of the covariance matrix.
    For dense covariance matrices, :math:`n(n+1)/2` parameters are added to 
    the model, storing the entries of the lower triangular portion of the
    Cholesky factorization of the covariance matrix.
    
    :param Model underlying_model: Model representing the likelihood with no
        random walk added.
    :param random_walk_idxs: A list or ``np.slice`` of 
        ``underlying_model`` model parameter indeces to add the random walk to.
        Indeces larger than ``underlying_model.n_modelparams`` should not 
        be touched.
    :param fixed_covariance: An ``np.ndarray`` specifying the fixed covariance 
        matrix (or diagonal thereof if ``diagonal`` is ``True``) of the 
        gaussian distribution. If set to ``None`` (default), this matrix is 
        presumed unknown and parameters are appended to the model describing 
        it.
    :param boolean diagonal: Whether the gaussian distribution covariance matrix
        is diagonal, or densely populated. Default is 
        ``True``.
    :param scale_mult: A function which takes an array of expparams and
        outputs a real number for each one, representing the scale of the 
        given experiment. This is useful if different experiments have 
        different time lengths and therefore incur different dispersion amounts.\
        If a string is given instead of a function, 
        thee scale multiplier is the ``exparam`` with that name.
    :param model_transformation: Either ``None`` or a pair of functions 
        ``(transform, inv_transform)`` specifying a transformation of ``modelparams``
        (of the underlying model) before gaussian noise is added, 
        and the inverse operation after
        the gaussian noise has been added.
    """
    def __init__(
            self, underlying_model, random_walk_idxs='all', 
            fixed_covariance=None, diagonal=True, 
            scale_mult=None, model_transformation=None
        ):
        
        self._diagonal = diagonal
        self._rw_idxs = np.s_[:underlying_model.n_modelparams] \
            if random_walk_idxs == 'all' else random_walk_idxs
            
        explicit_idxs = np.arange(underlying_model.n_modelparams)[self._rw_idxs]
        if explicit_idxs.size == 0:
            raise IndexError('At least one model parameter must take a random walk.')
    
        self._rw_names = [
                underlying_model.modelparam_names[idx] 
                for idx in explicit_idxs
            ]
        self._n_rw = len(explicit_idxs)
        
        self._srw_names = []
        if fixed_covariance is None:
            # In this case we need to lean the covariance parameters too,
            # therefore, we need to add modelparams
            self._has_fixed_covariance = False
            if self._diagonal:
                self._srw_names = ["\sigma_{{{}}}".format(name) for name in self._rw_names]
                self._srw_idxs = (underlying_model.n_modelparams + \
                    np.arange(self._n_rw)).astype(np.int)
            else:
                self._srw_idxs = (underlying_model.n_modelparams +
                    np.arange(self._n_rw * (self._n_rw + 1) / 2)).astype(np.int)
                # the following list of indeces tells us how to populate 
                # a cholesky matrix with a 1D list of values
                self._srw_tri_idxs = np.tril_indices(self._n_rw)
                for idx1, name1 in enumerate(self._rw_names):
                    for name2 in self._rw_names[:idx1+1]:
                        if name1 == name2:
                            self._srw_names.append("\sigma_{{{}}}".format(name1))
                        else:
                            self._srw_names.append("\sigma_{{{},{}}}".format(name2,name1))
        else:
            # In this case the covariance matrix is fixed and fully specified
            self._has_fixed_covariance = True
            if self._diagonal:
                if fixed_covariance.ndim != 1:
                    raise ValueError('Diagonal covariance requested, but fixed_covariance has {} dimensions.'.format(fixed_covariance.ndim))
                if fixed_covariance.size != self._n_rw:
                    raise ValueError('fixed_covariance dimension, {}, inconsistent with number of parameters, {}'.format(fixed_covariance.size, self.n_rw))
                self._fixed_scale = np.sqrt(fixed_covariance)
            else:
                if fixed_covariance.ndim != 2:
                    raise ValueError('Dense covariance requested, but fixed_covariance has {} dimensions.'.format(fixed_covariance.ndim))
                if fixed_covariance.size != self._n_rw **2 or fixed_covariance.shape[-2] != fixed_covariance.shape[-1]:
                    raise ValueError('fixed_covariance expected to be square with width {}'.format(self._n_rw))
                self._fixed_chol = np.linalg.cholesky(fixed_covariance)
                self._fixed_distribution = multivariate_normal(
                    np.zeros(self._n_rw),
                    np.dot(self._fixed_chol, self._fixed_chol.T)
                )
                
        super(GaussianRandomWalkModel, self).__init__(underlying_model)
        
        if np.max(np.arange(self.n_modelparams)[self._rw_idxs]) > np.max(explicit_idxs):
            raise IndexError('random_walk_idxs out of bounds; must index (a subset of ) underlying_model modelparams.')
            
        if scale_mult is None:
            self._scale_mult_fcn = (lambda expparams: 1)
        elif isinstance(scale_mult, basestring):
            self._scale_mult_fcn = lambda x: x[scale_mult]
        else:
            self._scale_mult_fcn = scale_mult
            
        self._has_transformation = model_transformation is not None
        if self._has_transformation:
            self._transform = model_transformation[0]
            self._inv_transform = model_transformation[1]
            
        
                
    ## PROPERTIES ##
    
    @property
    def modelparam_names(self):
        return self.underlying_model.modelparam_names + self._srw_names
        
    @property 
    def n_modelparams(self):
        return len(self.modelparam_names)
        
    @property
    def is_n_outcomes_constant(self):
        return False
            
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        ud_valid = self.underlying_model.are_models_valid(modelparams[...,:self.underlying_model.n_modelparams])
        if self._has_fixed_covariance:
            return ud_valid
        elif self._diagonal:
            pos_std = np.greater_equal(modelparams[...,self._srw_idxs], 0).all(axis=-1)
            return np.logical_and(ud_valid, pos_std)
        else:
            return ud_valid
    
    def likelihood(self, outcomes, modelparams, expparams):
        super(GaussianRandomWalkModel, self).likelihood(outcomes, modelparams, expparams)
        return self.underlying_model.likelihood(outcomes, modelparams[...,:self.underlying_model.n_modelparams], expparams)
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(GaussianRandomWalkModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams[...,:self.underlying_model.n_modelparams], expparams, repeat)
        
    def est_update_covariance(self, modelparams):
        """
        Returns the covariance of the gaussian noise process for one 
        unit step. In the case where the covariance is being learned,
        the expected covariance matrix is returned.
        
        :param modelparams: Shape `(n_models, n_modelparams)` shape array
        of model parameters.
        """
        if self._diagonal:
            cov = (self._fixed_scale ** 2 if self._has_fixed_covariance \
                else np.mean(modelparams[:, self._srw_idxs] ** 2, axis=0))
            cov = np.diag(cov)
        else:
            if self._has_fixed_covariance:
                cov = np.dot(self._fixed_chol, self._fixed_chol.T)
            else:
                chol = np.zeros((modelparams.shape[0], self._n_rw, self._n_rw))
                chol[(np.s_[:],) + self._srw_tri_idxs] = modelparams[:, self._srw_idxs]
                cov = np.mean(np.einsum('ijk,ilk->ijl', chol, chol), axis=0)
        return cov
        
    def update_timestep(self, modelparams, expparams):

        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]
        if self._diagonal:
            scale = self._fixed_scale if self._has_fixed_covariance else modelparams[:, self._srw_idxs]
            # the following works when _fixed_scale has shape (n_rw) or (n_mps,n_rw)
            # in the latter, each particle gets dispersed by its own belief of the scale
            steps = scale * np.random.normal(size = (n_eps, n_mps, self._n_rw))
            steps = steps.transpose((1,2,0))
        else:
            if self._has_fixed_covariance:
                steps = np.dot(
                    self._fixed_chol, 
                    np.random.normal(size = (self._n_rw, n_mps * n_eps))
                ).reshape(self._n_rw, n_mps, n_eps).transpose((1,0,2))
            else:
                chol = np.zeros((n_mps, self._n_rw, self._n_rw))
                chol[(np.s_[:],) + self._srw_tri_idxs] = modelparams[:, self._srw_idxs]
                # each particle gets dispersed by its own belief of the cholesky
                steps = np.einsum('kij,kjl->kil', chol, np.random.normal(size = (n_mps, self._n_rw, n_eps)))
        
        # multiply by the scales of the current experiments
        steps = self._scale_mult_fcn(expparams) * steps
        
        if self._has_transformation:
            # repeat model params for every expparam
            new_mps = np.repeat(modelparams[np.newaxis,:,:], n_eps, axis=0).reshape((n_eps * n_mps, -1))
            # run transformation on underlying slice
            new_mps[:, :self.underlying_model.n_modelparams] = self._transform(
                    new_mps[:, :self.underlying_model.n_modelparams]
                )
            # add on the random steps to the relevant indeces
            new_mps[:, self._rw_idxs] += steps.transpose((2,0,1)).reshape((n_eps * n_mps, -1))
            #  back to regular parameterization
            new_mps[:, :self.underlying_model.n_modelparams] = self._inv_transform(
                    new_mps[:, :self.underlying_model.n_modelparams]
                )
            new_mps = new_mps.reshape((n_eps, n_mps, -1)).transpose((1,2,0))
        else:
            new_mps = np.repeat(modelparams[:,:,np.newaxis], n_eps, axis=2)
            new_mps[:, self._rw_idxs, :] += steps

        return new_mps

## TESTING CODE ###############################################################

if __name__ == "__main__":
    
    import operator as op
    from .finite_test_models import SimplePrecessionModel
    
    m = BinomialModel(SimplePrecessionModel())
    
    os = np.array([6, 7, 8, 9, 10])
    mps = np.array([[0.1], [0.35], [0.77]])
    eps = np.array([(0.5 * np.pi, 10), (0.51 * np.pi, 10)], dtype=m.expparams_dtype)
    
    L = m.likelihood(
        os, mps, eps
    )
    print(L)
    
    assert m.call_count == reduce(op.mul, [os.shape[0], mps.shape[0], eps.shape[0]]), "Call count inaccurate."
    assert L.shape == (os.shape[0], mps.shape[0], eps.shape[0]), "Shape mismatch."
    
