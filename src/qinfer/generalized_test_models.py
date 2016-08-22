#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# finite_test_models.py: Simple models for testing inference engines 
#       where the number of outcomes is not finite.
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
    'BasicPoissonModel',
    'BasicGaussianModel',
    'ExponentialPoissonModel',
    'ExponentialGaussianModel'
]

## IMPORTS ###################################################################

from builtins import range

import numpy as np
from abc import ABCMeta,abstractmethod,abstractproperty
from scipy.special import gammaln
from .utils import binomial_pdf
from functools import wraps 
from .abstract_model import Model, DifferentiableModel
from .domains import IntegerDomain, RealDomain
    
## CLASSES ###################################################################

class PoissonModel(DifferentiableModel):
    r"""
    Abstract Poisson model class that describes a Poisson model with likelihood form 

    :math:`\Pr(k|f(\vec{x};\vec{c}))= \frac{f(\vec{x};\vec{c})^ke^{-f(\vec{x};\vec{c})}}{k!}`

    Where :math:`k` is the number of events observed, and :math:`f(\vec{x};\vec{c})`
    is the expected number of events observed given the model parameters :math:`\vec{x}` 
    and experimental parameters :math:`\vec{c}`.
    """

    __metaclass__ = ABCMeta
    
    ## INITIALIZER ##

    def __init__(self, num_outcome_samples=10000, allow_identical_outcomes=False):
        super(PoissonModel, self).__init__(allow_identical_outcomes=allow_identical_outcomes)
        self.num_outcome_samples = num_outcome_samples

        # domain is all non-negative integers
        self._domain = IntegerDomain(min=0, max=None)

    ## ABSTRACT METHODS##

    @abstractmethod
    def model_function(self,modelparams,expparams):
        """
        Return model function :math:`f(\vec{x};\vec{c})` specifying the expected 
        number of Poisson events observed for every combination of 
        model parameters :math:`\vec{x}` and experimental parameters :math:`\vec{c}`.

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A two-index tensor of shape ``(n_models, n_expparams)``.
        """
        pass

    @abstractmethod
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives :math:`\nabla_{\vec{x}}f(\vec{x};\vec{c})`
        in an array of shape ``(n_modelparams,n_models,n_expparams)``.

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
        array of model parameter vectors describing the hypotheses for
        which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A three-index tensor ``f[i, j,k]``, where ``i`` 
            indexes which model parameter the derivative was taken with respect to,
            ``j`` indexes which model is being considered, 
            and ``k`` indexes which experimental parameters was used.   
        """
        pass


    @abstractmethod
    def are_models_valid(self, modelparams):
        pass

    ## ABSTRACT PROPERTIES ##
    
    @abstractproperty
    def n_model_function_params(self):
        """
        Returns the number of real model function parameters admitted by this PoissonModel's,
        model function.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a :class:`Model` instance.       

        :rtype: int 
        """

    @abstractproperty
    def model_function_param_names(self):
        """
        Returns the names of the various model function parameters admitted by this
        model, formatted as LaTeX strings.    

        :rtype: list
        """


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
    def modelparam_names(self):
        return self.model_function_param_names

    @property
    def n_modelparams(self):
        return self.n_model_function_params   
    
    @property
    def is_outcomes_constant(self):
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

        super(PoissonModel, self).likelihood(outcomes, modelparams, expparams)

       
        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes = outcomes[...,np.newaxis, np.newaxis]
        else:
            outcomes = outcomes[:,np.newaxis,:]

        lamb_da = self.model_function(modelparams,expparams)[np.newaxis,...]
     
        return np.exp(outcomes*np.log(lamb_da)-gammaln(outcomes+1)-lamb_da)


    def score(self, outcomes, modelparams, expparams, return_L=False):
        
        
        super(PoissonModel, self).score(outcomes, modelparams, expparams, return_L) 

        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes_rs = outcomes[np.newaxis,...,np.newaxis, np.newaxis]
        else:
            outcomes_rs = outcomes[np.newaxis,:,np.newaxis,:]

        lamb_da = self.model_function(modelparams,expparams)[np.newaxis,np.newaxis,:,:]
        fns_deriv = self.model_function_derivative(modelparams,expparams)[:,np.newaxis,:,:]


        scr = (outcomes_rs/lamb_da-1)*fns_deriv
        
        if return_L:
            return scr, self.likelihood(outcomes, modelparams, expparams)
        else:
            return scr


    def simulate_experiment(self,modelparams,expparams,repeat=1):

        super(PoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        if modelparams.ndim == 1:
            modelparams = modelparams[np.newaxis, ...]   
        
        lamb_das = self.model_function(modelparams,expparams)
        outcomes = np.asarray(np.random.poisson(np.tile(lamb_das[np.newaxis,...],(repeat,1,1)))
                    ).reshape(repeat,modelparams.shape[0],expparams.shape[0]).astype(self.domain(None).dtype)

        return (outcomes[0, 0, 0] if repeat == 1 and expparams.shape[0] == 1 and modelparams.shape[0] == 1 else outcomes
                ).astype(self.domain(None).dtype)
        

class BasicPoissonModel(PoissonModel):
    """
    The basic Poisson model consisting of a single model parameter :math:`\lambda`,
    describing an event rate and a single experimental parameter :math:`\tau` describing 
    how long we measure the event rate for.
    """
    @property 
    def n_model_function_params(self):
        return 1

    def model_function(self,modelparams,expparams):
        """
        Return model functions in form [idx_modelparams,idx_expparams]. The model function 
        therefore returns the plain model parameters, but tiles them over the number of experiments 
        to satisfy the requirements of the abstract method. The shape of `expparams` therefore signifies 
        the number of experiments that will be performed.
        """
        return modelparams.flatten()[:,np.newaxis] * expparams['tau'][np.newaxis,:]
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_model,idx_expparams]
        """
        return np.tile(expparams['tau'], (modelparams.shape[0],1))[np.newaxis,...]
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)
  
    @property
    def model_function_param_names(self):
        return [r'\lambda']
    
    @property
    def expparams_dtype(self):
        return [('tau', 'float')]

class ExponentialPoissonModel(PoissonModel):
    """
    A Poisson model with an exponential growth model function
    consisting of a single model parameter :math:`T1` controlling the
    rate and a single experimental parameter :math:`\tau`.
    """

    def __init__(self, max_rate=100, num_outcome_samples=10000, allow_identical_outcomes=False):

        super(ExponentialPoissonModel, self).__init__(num_outcome_samples=num_outcome_samples, 
            allow_identical_outcomes=allow_identical_outcomes)
        self.max_rate = max_rate

    @property 
    def n_model_function_params(self):
        return 1

    def model_function(self,modelparams,expparams):
        """
        Return model functions in form [idx_expparams,idx_modelparams]. The model function 
        therefore returns the plain model parameters, but tiles them over the number of experiments 
        to satisfy the requirements of the abstract method. The shape of `expparams` therefore signifies 
        the number of experiments that will be performed.
        """

        return self.max_rate*(1-np.exp(-expparams['tau']/modelparams))
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_expparams,idx_modelparams]
        """

        return -self.max_rate*(expparams['tau']/modelparams**2)*np.exp(-expparams['tau']/modelparams)

    def are_models_valid(self, modelparams):
        return np.logical_not(np.any(modelparams<0,axis=1))
    
    @property
    def model_function_param_names(self):
        return [r'T1']
    
    @property
    def expparams_dtype(self):
        return [('tau','float')]

class GaussianModel(DifferentiableModel):
    r"""
    Abstract Gaussian model class that describes a Gaussian model with likelihood form 

    :math:`\Pr(y|\mu,\var,f(\vec{x};\vec{c}))= \frac{1}{\sqrt{2\var\pi}}e^{-\frac{(x-\mu)^2}{2\var}}`

    Where :math:`y` is the observed outcome, and :math:`f(\vec{x};\vec{c})`
    is some underlying model function with unknown parameters :math:`\vec{x}` and experimental 
    parameters :math:`\vec{c}`. The distribution is defined by the mean :math:`\mu` and the variance,
    :math:`\var`. These may be either unknown model parameters to be learned, or fixed parameters. 

    Can optionally add model parameters for unknown :math:`\mu`, and :math:`\var` you should not use these
    as modelparameter names. 
    """

    __metaclass__ = ABCMeta
    
    ## INITIALIZER ##

    def __init__(self, var=None, num_outcome_samples=10000,constant_noise_outcomes=False):

        self.num_outcome_samples = num_outcome_samples
        self._var = var
        self._constant_noise_outcomes = constant_noise_outcomes
        super(GaussianModel, self).__init__(allow_identical_outcomes=True)

        # The domain is always the set of all real numbers
        self._domain = RealDomain(min=None, max=None)

    ## ABSTRACT METHODS##

    @abstractmethod
    def model_function(self,modelparams,expparams):
        """
        Return model function :math:`f(\vec{x};\vec{c})` with unknown parameters :math:`\vec{x}` 
        and experimental parameters :math:`\vec{c}` in the form [idx_modelparams,idx_expparams].

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A two-index tensor ``f[i, j]``, where ``i`` indexes which model parameters are
            being considered, ``j`` indexes which experimental parameters was used.   
        """
        pass

    @abstractmethod
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives :math:`\nabla_{\vec{x}}f(\vec{x};\vec{c})`
        in form [idx_modelparam,idx_model,idx_expparam].

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
        array of model parameter vectors describing the hypotheses for
        which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A three-index tensor ``f[i, j,k]``, where ``i`` indexes which model parameter the derivative was taken with respect to,
            ``j`` indexes which model is being considered, 
            and ``k`` indexes which experimental parameters was used.   
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


    @abstractproperty
    def n_model_function_params(self):
        """
        Returns the number of real model function parameters admitted by this GaussianModel's,
        model function.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a :class:`Model` instance.       

        :rtype: int 
        """

    @abstractproperty
    def model_function_param_names(self):
        """
        Returns the names of the various model function parameters admitted by this
        model, formatted as LaTeX strings.    

        :rtype: list
        """
    ## PROPERTIES ##
    
    
    @property
    def is_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        """
        return True

    
    @property
    def modelparam_names(self):
        if self._var is None:
            return self.model_function_param_names+[r'\var']
        else:
            return self.model_function_param_names

    @property
    def n_modelparams(self):
        if self._var is None:
            return self.n_model_function_params+1
        else:
            return self.n_model_function_params
    
    
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

        super(GaussianModel, self).likelihood(outcomes, modelparams, expparams)

        
        if modelparams.ndim == 1:
            modelparams = modelparams[np.newaxis, ...]
      
        if len(outcomes.shape) == 1:
            outcomes = outcomes[...,np.newaxis, np.newaxis]
        else:
            outcomes = outcomes[:,np.newaxis,:]

        # Check to see if var/mu are model parameters, and if 
        # so remove from model parameter array 
        if self._var is None:
            var_index = self.modelparam_names.index(r'\var')
            var = modelparams[:,var_index][np.newaxis,:,np.newaxis]
            modelparams = np.delete(modelparams,var_index,1)
        else: 
            var = np.full((1,modelparams.shape[0],1),self._var)

        x = self.model_function(modelparams,expparams)

        return 1/(np.sqrt(2*np.pi*var))*np.exp(-(outcomes-x)**2/(2*var))


    def score(self, outcomes, modelparams, expparams, return_L=False):
        
        super(GaussianModel, self).score(outcomes, modelparams, expparams, return_L) 

        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes_rs = outcomes[np.newaxis,...,np.newaxis, np.newaxis]
        else:
            outcomes_rs = outcomes[np.newaxis,:,np.newaxis,:]


        if self._var is None:
            var_index = self.modelparam_names.index(r'\var')
            var = modelparams[:,var_index][np.newaxis,:,np.newaxis]
            modelparams_rs = np.delete(modelparams,var_index,1)
        else: 
            var = np.empty((1,modelparams.shape[0],1))
            var[...] = self._var
            modelparams_rs = modelparams


        x = self.model_function(modelparams_rs,expparams)[np.newaxis,np.newaxis,:,:]
        fns_deriv = self.model_function_derivative(modelparams_rs,expparams)[:,np.newaxis,:,:]
        
   
        scr = ((outcomes_rs-x)/var)*fns_deriv

  
        # make room in array for var and mu derivatives
        scr = np.pad(scr,((0,self.n_modelparams-scr.shape[0]),(0,0),(0,0),(0,0)),mode='constant',constant_values=0)
        
        if self._var is None:
            scr[var_index] = (outcomes_rs-x)**2/(2*np.power(var,2)) - 1/(2*var)

        if return_L:
            return scr, self.likelihood(outcomes, modelparams, expparams)
        else:
            return scr


    def simulate_experiment(self,modelparams,expparams,repeat=1):

        super(GaussianModel, self).simulate_experiment(modelparams, expparams, repeat)

        if modelparams.ndim == 1:
            modelparams = modelparams[np.newaxis, ...]  
        
        if expparams.ndim == 1:
            expparams = expparams[..., np.newaxis]
        
        if self._var is None:
            var_index = self.modelparam_names.index(r'\var')
            var = modelparams[:,var_index][np.newaxis,:,np.newaxis]
            modelparams = np.delete(modelparams,var_index,1)
        else: 
            var = (self._var * np.ones(modelparams.shape[0]))[np.newaxis,:,np.newaxis]
        
        x = self.model_function(modelparams,expparams)
        x = np.tile(x, (repeat, 1, 1))
        var = np.tile(var, (repeat, 1,expparams.shape[0]))

        outcomes = np.random.normal(x, var).astype(self.domain(None).dtype)

        return outcomes[0, 0, 0] if repeat == 1 and expparams.shape[0] == 1 and modelparams.shape[0] == 1 else outcomes
                



class BasicGaussianModel(GaussianModel):
    """
    The basic Gaussian model consisting of a single model parameter :math:`\lambda`,
    and a single experiment parameter :math:`\tau`: which corresponds to a linear model
    function :math:`f(\tau)=\lambda\tau`.
    """

    @property 
    def n_model_function_params(self):
        return 1

    def model_function(self,modelparams,expparams):
        """
        Return model functions in form [idx_modelparams,idx_expparams]. The model function 
        therefore returns the plain model parameters, but tiles them over the number of experiments 
        to satisfy the requirements of the abstract method. The shape of `expparams` therefore signifies 
        the number of experiments that will be performed.
        """
        return modelparams.flatten()[:,np.newaxis] * expparams['tau'][np.newaxis,:]
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_model,idx_expparams]
        """
        return np.tile(expparams['tau'], (modelparams.shape[0],1))[np.newaxis,...]
    
    def are_models_valid(self, modelparams):
        return np.ones(modelparams.shape[0],dtype=bool)
    
    @property
    def model_function_param_names(self):
        return [r'\lambda']
    
    @property
    def expparams_dtype(self):
        return [('tau', 'float')]

class ExponentialGaussianModel(GaussianModel):
    """
    A Poisson model with an exponential growth model function
    consisting of a single model parameter :math:`T1` controlling the
    rate and a single experimental parameter :math:`\tau`.
    """

    @property 
    def n_model_function_params(self):
        return 1

    def model_function(self,modelparams,expparams):
        """
        Return model functions in form [idx_expparams,idx_modelparams]. The model function 
        therefore returns the plain model parameters, but tiles them over the number of experiments 
        to satisfy the requirements of the abstract method. The shape of `expparams` therefore signifies 
        the number of experiments that will be performed.
        """
        # Note that this function does _not_ get passed var if it is a modelparam
        result = 1-np.exp(-expparams['tau'].T/np.tile(modelparams, expparams.shape[0]))
        return result
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_expparams,idx_modelparams]
        """
        # Note that this function does _not_ get passed var if it is a modelparam
        eps = expparams['tau'].T
        mps = np.tile(modelparams, expparams.shape[0])
        return (-(eps / mps**2) * np.exp(-eps / mps))[np.newaxis, :]

    def are_models_valid(self, modelparams):
        return np.logical_not(np.any(modelparams<0,axis=1))
    
    @property
    def model_function_param_names(self):
        return [r'T1']
    
    @property
    def expparams_dtype(self):
        return [('tau','float')]

