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

    def __init__(self, num_outcome_samples=10000):
        super(PoissonModel, self).__init__()
        self.num_outcome_samples = num_outcome_samples

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
        in form [idx_modelparam,idx_modelparams,idx_expparams].

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
        array of model parameter vectors describing the hypotheses for
        which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
        :rtype: np.ndarray
        :return: A three-index tensor ``f[i, j,k]``, where ``i`` indexes which model parameter the derivative was taken with respect to,
            ``j`` indexes which model parameters are being considered, 
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
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes = outcomes[..., np.newaxis]
        lamb_da = self.model_function(modelparams,expparams)[np.newaxis,...]
        outcomes = outcomes[:,np.newaxis,:]
        return np.exp(outcomes*np.log(lamb_da)-gammaln(outcomes+1)-lamb_da)


    def score(self, outcomes, modelparams, expparams, return_L=False):
        
        
        super(PoissonModel, self).score(outcomes, modelparams, expparams, return_L) 

        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes = outcomes[..., np.newaxis]

        lamb_da = self.model_function(modelparams,expparams)[np.newaxis,np.newaxis,:,:]
        fns_deriv = self.model_function_derivative(modelparams,expparams)[:,np.newaxis,:,:]
        outcomes_rs = outcomes[np.newaxis,:,np.newaxis,:]

        scr = (outcomes_rs/lamb_da-1)*fns_deriv
        
        if return_L:
            return scr, self.likelihood(outcomes, modelparams, expparams)
        else:
            return scr


    def simulate_experiment(self,modelparams,expparams,repeat=1):

        super(PoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, :]    
        
        lamb_das = self.model_function(modelparams,expparams)
        outcomes = np.asarray(np.random.poisson(np.tile(lamb_das[np.newaxis,...],(repeat,1,1)))
                    ).reshape(repeat,modelparams.shape[0],expparams.shape[0]).astype(self.outcomes_dtype)

        return outcomes 


class BasicPoissonModel(PoissonModel):
    """
    The basic Poisson model consisting of a single model parameter :math:`\lambda`,
    and no experimental parameters.
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
        return np.tile(modelparams,expparams.shape[0])
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_expparams,idx_modelparams]
        """
        return np.ones((1,modelparams.shape[0],expparams.shape[0]))


    
    def are_models_valid(self, modelparams):
        return np.all(modelparams >= 0, axis=1)

    ## ABSTRACT PROPERTIES ##
    @property
    def model_function_param_names(self):
        return [r'\lambda']
    
    @property
    def expparams_dtype(self):
        []

class ExponentialPoissonModel(PoissonModel):
    """
    A Poisson model with an exponential growth model function
    consisting of a single model parameter :math:`T1` controlling the
    rate and a single experimental parameter :math:`\tau`.
    """

    def __init__(self,max_rate=100, num_outcome_samples=10000):
        super(ExponentialPoissonModel, self).__init__(num_outcome_samples=num_outcome_samples)
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

    ## ABSTRACT PROPERTIES ##
    
    @property
    def model_function_param_names(self):
        return [r'T1']
    
    @property
    def expparams_dtype(self):
        return [('tau','float')]

class GaussianModel(DifferentiableModel):
    r"""
    Abstract Gaussian model class that describes a Gaussian model with likelihood form 

    :math:`\Pr(y|\mu,\sigma,f(\vec{x};\vec{c}))= \frac{1}{\sqrt{2\sigma^2\pi}}e^(-\frac{(x-\mu)^2}{2\sigma^2}`

    Where :math:`y` is the observed outcome, and :math:`f(\vec{x};\vec{c})`
    is some underlying model function with unknown parameters :math:`\vec{x}` and experimental 
    parameters :math:`\vec{c}`. The distribution is defined by the mean :math:`\mu` and the variance,
    :math:`\sigma^2`. These may be either unknown model parameters to be learned, or fixed parameters. 

    Can optionally add model parameters for unknown :math:`\mu`, and :math:`\sigma` you should not use these
    as modelparameter names. 
    """

    __metaclass__ = ABCMeta
    
    ## INITIALIZER ##

    def __init__(self, sigma=None, num_outcome_samples=10000):

        self.num_outcome_samples = num_outcome_samples
        self._sigma = sigma

        super(GaussianModel, self).__init__()


    

        

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
            ``j`` indexes which model parameters are being considered, 
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
    def outcomes_dtype(self):
        return 'float32'
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        """
        return True

    
    @property
    def modelparam_names(self):
        if self._sigma is None:
            return self.model_function_param_names+[r'\sigma']
        else:
            return self.model_function_param_names

    @property
    def n_modelparams(self):
        if self._sigma is None:
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
    

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.

        super(GaussianModel, self).likelihood(outcomes, modelparams, expparams)

       
        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes = outcomes[..., np.newaxis]

        # Check to see if sigma/mu are model parameters, and if 
        # so remove from model parameter array 
        if self._sigma is None:
            sigma_index = self.modelparam_names.index(r'\sigma')
            sigma = modelparams[:,sigma_index][np.newaxis,:,np.newaxis]
            modelparams = np.delete(modelparams,sigma_index,1)
        else: 
            sigma = np.empty((1,modelparams.shape[0],1))
            sigma[...] = self._sigma


        x = self.model_function(modelparams,expparams)[np.newaxis,...]
        outcomes = outcomes[:,np.newaxis,:]
        
        return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(outcomes-x)**2/(2*sigma**2))


    def score(self, outcomes, modelparams, expparams, return_L=False):
        
        super(GaussianModel, self).score(outcomes, modelparams, expparams, return_L) 

        if len(modelparams.shape) == 1:
            modelparams = modelparams[np.newaxis, ...]
        
        if len(outcomes.shape) == 1:
            outcomes = outcomes[..., np.newaxis]


        if self._sigma is None:
            sigma_index = self.modelparam_names.index(r'\sigma')
            sigma = modelparams[:,sigma_index][np.newaxis,:,np.newaxis]
            modelparams_rs = np.delete(modelparams,sigma_index,1)
        else: 
            sigma = np.empty((1,modelparams.shape[0],1))
            sigma[...] = self._sigma
            modelparams_rs = modelparams


        x = self.model_function(modelparams_rs,expparams)[np.newaxis,np.newaxis,:,:]
        fns_deriv = self.model_function_derivative(modelparams_rs,expparams)[:,np.newaxis,:,:]
        
        outcomes_rs = outcomes[np.newaxis,:,np.newaxis,:]
   
        scr = ((outcomes_rs-x)/sigma**2)*fns_deriv

  
        # make room in array for sigma and mu derivatives
        scr = np.pad(scr,((0,self.n_modelparams-scr.shape[0]),(0,0),(0,0),(0,0)),mode='constant',constant_values=0)
        
        if self._sigma is None:
            scr[sigma_index] = (outcomes_rs-x)**2/np.power(sigma,3) - 1/sigma


        
        
        
        
        if return_L:
            return scr, self.likelihood(outcomes, modelparams, expparams)
        else:
            return scr


    def simulate_experiment(self,modelparams,expparams,repeat=1):

        super(GaussianModel, self).simulate_experiment(modelparams, expparams, repeat)

        if len(modelparams.shape) == 1:
            modelparams = modelparams = modelparams[np.newaxis, ...]  
        
        if len(expparams.shape) == 1:
            expparams = expparams[..., np.newaxis]
        
        if self._sigma is None:
            sigma_index = self.modelparam_names.index(r'\sigma')
            sigma = modelparams[:,sigma_index]
            modelparams = np.delete(modelparams,sigma_index,1)
        else: 
            sigma = np.empty(modelparams.shape[0])
            sigma[...] = self._sigma

       
        x = self.model_function(modelparams,expparams)
        outcomes = np.asarray(np.random.normal(x,np.tile(sigma[np.newaxis,:,np.newaxis],(repeat,1,1)))).reshape(
            repeat,modelparams.shape[0],expparams.shape[0]).astype(self.outcomes_dtype)
        return outcomes 



class BasicGaussianModel(GaussianModel):
    """
    The basic Gaussian model consisting of a single model parameter :math:`\mu`,
    and no experimental parameters.
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
        return np.tile(modelparams,expparams.shape[0])
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_expparams,idx_modelparams]
        """
        return np.ones((1,modelparams.shape[0],expparams.shape[0]))


    
    def are_models_valid(self, modelparams):
        return np.ones(modelparams.shape[0],dtype=bool)

    ## ABSTRACT PROPERTIES ##
    
    @property
    def model_function_param_names(self):
        return [r'\mu']
    
    @property
    def expparams_dtype(self):
        []

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
   
        return 1-np.exp(-expparams['tau']/modelparams)
    
    def model_function_derivative(self,modelparams,expparams):
        """
        Return model functions derivatives in form [idx_modelparam,idx_expparams,idx_modelparams]
        """

        return -(expparams['tau']/modelparams**2)*np.exp(-expparams['tau']/modelparams)

    def are_models_valid(self, modelparams):
        return np.logical_not(np.any(modelparams<0,axis=1))

    ## ABSTRACT PROPERTIES ##
    
    @property
    def model_function_param_names(self):
        return [r'T1']
    
    @property
    def expparams_dtype(self):
        return [('tau','float')]

