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
from abc import ABCMeta,abstractmethod,abstractproperty
from scipy.special import gammaln
from .utils import binomial_pdf

from .abstract_model import Model, DifferentiableModel
    
## CLASSES ###################################################################

class PoissonModel(DifferentiableModel):
    r"""
    Abstract Poisson model class that describes a Poisson model with likelihood form 

    :math:`\Pr(k|f(\vec{x};\vec{c}))= \frac{f(\vec{x};\vec{c})^ke^{-f(\vec{x};\vec{c})}}{k!}`

    Where :math:`k` is the number of outcomes observed per unit time, and :math:`f(\vec{x};\vec{c})`
    is some underlying rate function with unknown parameters :math:`\vec{x}` and experimental 
    parameters :math:`\vec{c}`.
    """

    __metaclass__ = ABCMeta
    
    ## INITIALIZER ##

    def __init__(self, num_outcome_samples=500):
        super(PoissonModel, self).__init__()
        self.num_outcome_samples = num_outcome_samples

    ## ABSTRACT METHODS##

    @abstractmethod
    def model_function(self,modelparams,expparams):
        """
        Return model function :math:`f(\vec{x};\vec{c})` with unknown parameters :math:`\vec{x}` 
        and experimental parameters :math:`\vec{c}` in the form [idx_expparams,idx_modelparams].

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A two-index tensor ``f[i, j]``, where ``i`` indexes which experimental parameters are
            being considered, ``j`` indexes which vector of model parameters was used.   
        """
        pass

    @abstractmethod
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives :math:`\nabla_{\vec{x}}f(\vec{x};\vec{c})`
        in form [idx_modelparam,idx_expparams,idx_modelparams].

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
        array of model parameter vectors describing the hypotheses for
        which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A three-index tensor ``f[i, j,k]``, where ``i`` indexes which model parameter the derivative was taken with respect to,
            ``j`` indexes which experimental parameters are being considered, 
            and ``k`` indexes which vector of model parameters was used.   
        """
        pass


    @abstractmethod
    def are_models_valid(self, modelparams):
        pass

    ## ABSTRACT PROPERTIES ##
    
    @abstractproperty
    def modelparam_names(self):
        """
        Returns the names of the various model parameters admitted by this
        model, formatted as LaTeX strings.
        """
        pass


    @abstractproperty
    def expparams_dtype(self):
        """
        Returns the dtype of an experiment parameter array. For a
        model with single-parameter control, this will likely be a scalar dtype,
        such as ``"float64"``. More generally, this can be an example of a
        record type, such as ``[('time', 'float64'), ('axis', 'uint8')]``.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        pass
    ## PROPERTIES ##
    
  
        
    
    
    @property
    def outcomes_dtype(self):
        return 'uint32'
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        """
        return True
    
    ## METHODS ##
    

    
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

        lamb_da = self.model_function(modelparams,expparams)[np.newaxis,...]
        outcomes = outcomes[:,np.newaxis,:]
        return np.exp(outcomes*np.log(lamb_da)-gammaln(outcomes+1)-lamb_da)


    def score(self, outcomes, modelparams, expparams, return_L=False):
        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]
        
        return super(PoissonModel, self).score(outcomes, modelparams, expparams, return_L) 

        outcomes_reshaped = outcomes[np.newaxis,:,np.newaxis,np.newaxis]
        modelparams_reshaped = modelparams[:,np.newaxix,:,:]

        fns_deriv = self.model_function_derivative(modelparams,expparams)[:,np.newaxis,:,:]
        
        scr = (outcomes/lamb_da-1)*fns_deriv
        
        if return_L:
            return scr, self.likelihood(outcomes, modelparams, expparams)
        else:
            return scr


    def simulate_experiment(self,modelparams,expparams,repeat=1):

        super(PoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]    
        
        lamb_das = self.model_function(modelparams,expparams)
        outcomes = np.random.poisson(lamb_das)

        return outcomes 


class BasicPoissonModel(PoissonModel):
    """
    The basic Poisson model consisting of a single model parameter :math:`\lambda`,
    and no experiment parameters.
    """
    @abstractmethod
    def model_function(self,modelparams,expparams):
        """
        Return model functions in form [idx_expparams,idx_modelparams]. The model function 
        therefore returns the plain model parameters, but tiles them over the number of experiments 
        to satisfy the requirements of the abstract method. The shape of `expparams` therefore signifies 
        the number of experiments that will be performed.
        """
        return np.tile(modelparams,expparams.shape[0]).transpose()

    @abstractmethod
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_expparams,idx_modelparams]
        """
        return np.ones(1,expparams.shape[0],modelparams.shape[0])


    @abstractmethod
    def are_models_valid(self, modelparams):
        pass

    ## ABSTRACT PROPERTIES ##
    
    @abstractproperty
    def modelparam_names(self):
        pass


    @abstractproperty
    def expparams_dtype(self):
        pass

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



    
        
