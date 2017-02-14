#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# expdesign.py: Adaptive experimental design algorithms.
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
from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ExperimentDesigner',
    'Heuristic',
    'EnsembleHeuristic',
    'ExpSparseHeuristic',
    'PGH',
    'OptimizationAlgorithms',
    'UtilityFunctions',
    'Bounds',
    'RandomDisplacementBounds'
]

## IMPORTS ####################################################################

from future.utils import with_metaclass

import numpy as np

# for BCRB and BED classes
import scipy.optimize as opt
from qinfer._lib import enum # <- TODO: replace with flufl.enum!

from abc import ABCMeta, abstractmethod
import warnings

from qinfer.finite_difference import *

## FUNCTIONS ###################################################################

def identity(arg): return arg

## CLASSES #####################################################################

OptimizationAlgorithms = enum.enum("NULL", "CG", "NCG", "NELDER_MEAD","LINEAR_SWEEP","RANDOM_GUESSES",
    "L_BFGS_B","BASIN_HOPPING")

UtilityFunctions = enum.enum('RISK','RISK_IMPROVEMENT',"INFORMATION_GAIN")

class Heuristic(with_metaclass(ABCMeta, object)):
    r"""
    Defines a heuristic used for selecting new experiments without explicit
    optimization of the risk. As an example, the :math:`t_k = (9/8)^k`
    heuristic discussed by [FGC12]_ does not make explicit reference to the
    risk, and so would be appropriate as a `Heuristic` subclass.
    In particular, the [FGC12]_ heuristic is implemented by the
    :class:`ExpSparseHeuristic` class.
    """

    def __init__(self, updater):
        self._updater = updater
    
    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError("Not yet implemented.")

class EnsembleHeuristic(Heuristic):
    r"""
    Heuristic that randomly chooses one of several other
    heuristics.

    :param list ensemble: List of tuples ``(heuristic, pr)``
        specifying the probability of choosing each member
        heuristic.
    """

    def __init__(self, ensemble):
        self._pr = np.array([pr for heuristic, pr in ensemble])
        self._heuristics = ([heuristic for heuristic, pr in ensemble])

    def __call__(self, *args):
        idx_heuristic = np.random.choice(len(self._heuristics), p=self._pr)
        return self._heuristics[idx_heuristic](*args)
        
class ExpSparseHeuristic(Heuristic):
    r"""
    Implements the exponentially-sparse time evolution heuristic
    of [FGC12]_, under which :math:`t_k = A b^k`, where :math:`A`
    and :math:`b` are parameters of the heuristic.

    :param qinfer.smc.SMCUpdater updater: Posterior updater for which
        experiments should be heuristicly designed.
    :param float scale: The value of :math:`A`, implicitly setting
        the frequency scale for the problem.
    :param float base: The base of the exponent; in general, should
        be closer to 1 for higher-dimensional models.
    :param str t_field: Name of the expparams field representing time.
        If None, then the generated expparams are taken to be scalar,
        and not a record.
    :param dict other_fields: Values of the other fields to be used
        in designed experiments.
    """

    def __init__(self,
            updater, scale=1, base=9/8,
            t_field=None, other_fields=None
        ):
        super(ExpSparseHeuristic, self).__init__(updater)
        self._scale = scale
        self._base = base
        self._t_field = t_field
        self._other_fields = other_fields

    def __call__(self):
        n_exps = len(self._updater.data_record)
        t = self._scale * (self._base ** n_exps)
        dtype = self._updater.model.expparams_dtype

        if self._t_field is None:
            return np.array([t], dtype=dtype)
        else:
            eps = np.empty((1,), dtype=dtype)
            for field, value in self._other_fields.items():
                eps[field] = value
            eps[self._t_field] = t
            return eps

class PGH(Heuristic):
    """
    Implements the *particle guess heuristic* (PGH) of [WGFC13a]_, which
    selects two particles from the current posterior, selects one as an
    inversion hypothesis and sets the time parameter to be the inverse of
    the distance between the particles. In this way, the PGH adapts to the
    current uncertianty without additional simulation resources.
    
    :param qinfer.smc.SMCUpdater updater: Posterior updater for which
        experiments should be heuristicly designed.
    :param str inv_field: Name of the ``expparams`` field corresponding to the
        inversion hypothesis.
    :param str t_field: Name of the ``expparams`` field corresponding to the
        evolution time.
    :param callable inv_func: Function to be applied to modelparameter vectors
        to produce an inversion field ``x_``.
    :param callable t_func: Function to be applied to the evolution time to produce a
         time field ``t``.
    :param int maxiters: Number of times to try and choose distinct particles
        before giving up.
    :param dict other_fields: Values to set for fields not given by the PGH.
    
    Once initialized, a ``PGH`` object can be called to generate a new
    experiment parameter vector:
    
    >>> pgh = PGH(updater) # doctest: +SKIP
    >>> expparams = pgh() # doctest: +SKIP
    
    If the posterior weights are very highly peaked (that is, if the effective
    sample size is too small, as measured by
    :attr:`~qinfer.smc.SMCUpdater.n_ess`), then it may be the case that the two
    particles chosen by the PGH are identical, such that the time would be
    determined by ``1 / 0``. In this case, the `PGH` class will instead draw
    new pairs of particles until they are not identical, up to ``maxiters``
    attempts. If that limit is reached, a `RuntimeError` will be raised.
    """
    
    def __init__(self, updater, inv_field='x_', t_field='t',
                 inv_func=identity,
                 t_func=identity,
                 maxiters=10,
                 other_fields=None
                 ):
        super(PGH, self).__init__(updater)
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        
    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))
            
        eps = np.empty((1,), dtype=self._updater.model.expparams_dtype)
        eps[self._x_] = self._inv_func(x)
        eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp))
        
        for field, value in self._other_fields.items():
            eps[field] = value
        
        return eps

class ExperimentDesigner(object):
    """
    Designs new experiments using the current best information provided by a
    Bayesian updater.
    
    :param qinfer.smc.SMCUpdater updater: A Bayesian updater to design
        experiments for.
    :param OptimizationAlgorithms opt_algo: Algorithm to be used to perform
        local optimization.
    """

    

    def __init__(self, updater,utility_func=None, opt_algo=OptimizationAlgorithms.CG):
        if opt_algo not in OptimizationAlgorithms.reverse_mapping:
            raise ValueError("Unsupported or unknown optimization algorithm.")

        self._updater = updater
        if utility_func is None or (utility_func==UtilityFunctions.RISK_IMPROVEMENT):
            self._utility_func = self.risk_improvement_utility
        elif utility_func==UtilityFunctions.RISK :
            self._utility_func = self.risk_utility
        elif utility_func==UtilityFunctions.INFORMATION_GAIN:
            self._utility_func = self.information_gain_utility
        else:
            self._utility_func = utility_func
            
        self._opt_algo = opt_algo
        
        # Set everything up for the first experiment.
        self.new_exp()
        self._num_calls = 0
        
    ## METHODS ################################################################
    def risk_utility(self,*args,**kwargs):
        return self._updater.bayes_risk(*args,use_cached_samples=True,cache_samples=True,**kwargs)
    
    def risk_improvement_utility(self,*args,**kwargs):
        return self._updater.risk_improvement(*args,use_cached_samples=True,cache_samples=True,**kwargs)

    def information_gain_utility(self,*args,**kwargs):
        return -self._updater.expected_information_gain(*args,use_cached_samples=True,cache_samples=True,**kwargs)

    def new_exp(self):
        """
        Resets this `ExperimentDesigner` instance and prepares for designing
        the next experiment.
        """
        self.__best_cost = None
        self.__best_ep = None
        
    def design_expparams_field(self,
            guess, fields,n_exps=1,
            cost_scale_k=1.0, disp=False,
            maxiter=None, maxfun=None,
            store_guess=False, grad_h=None, cost_mult=False,
            n_samples=None,
            bounds=None,niter=100,**opt_options):
        r"""
        Designs a new experiment by varying a field (or multiple) of a shape ``(n,)``
        record array and minimizing the objective function
        
        .. math::
            O(\vec{e}) = r(\vec{e}) + k \$(\vec{e}),
        
        where :math:`r` is the Bayes risk as calculated by the updater, and
        where :math:`\$` is the cost function specified by the model. Here,
        :math:`k` is a parameter specified to relate the units of the risk and
        the cost. See :ref:`expdesign` for more details.
        
        :param guess: Either a record array with a single guess,an array of guesses
            for SWEEP_GUESS or a callable function that generates guesses.
        :type guess: Instance of :class:`~Heuristic`, `callable`
            or :class:`~numpy.ndarray` of ``dtype``
            :attr:`~qinfer.abstract_model.Simulatable.expparams_dtype`
        :param str list fields: The names of the ``expparams`` fields to be optimized,
            eg. of the form 'field' for one field, and ['field1','field2',...] for 
            multiple fields. All other fields of ``guess`` will be held constant.
        :param float cost_scale_k: A scale parameter :math:`k` relating the
            Bayes risk to the experiment cost.
            See :ref:`expdesign`.
        :param bool disp: If `True`, the optimization will print additional
            information as it proceeds.
        :param int maxiter: For those optimization algorithms which support
            it (currently, only CG and NELDER_MEAD), limits the number of
            optimization iterations used for each guess.
        :param int maxfun: For those optimization algorithms which support it
            (currently, only NCG and NELDER_MEAD), limits the number of
            objective calls that can be made.
        :param bool store_guess: If ``True``, will compare the outcome of this
            guess to previous guesses and then either store the optimization of
            this experiment, or the previous best-known experiment design.
        :param float grad_h: Step size to use in estimating gradients. Used
            only if ``opt_algo`` is NCG.
        :param dict opt_options: Dictionary of additional keyword arguments to
            be passed to the optimization function.
        :param list bounds: Bounds for the optimized parameter of the form (min,max).
             Used only if ``opt_algo`` is L_BFGS_B, BASIN_HOPPING, LINEAR_SWEEP or 
             RANDOM_GUESSES.
        :param int niter: Number of iterations for optimization algorithms that
            take a fixed amount of iterations. Currently only used if ``opt_algo`` is 
            BASIN_HOPPING. Bounds must be set for these optimization algorithms. 
        :param n_samples:  Number of random guesses or linear parameter sweep to generate.
            Used only if ``opt_algo`` is LINEAR_SWEEP and RANDOM_GUESSES
            Elements must be of type ``dtype``
            
        :return: An array representing the best experiment design found so
            far for the current experiment.
        """
        # Check if a single field is passed and convert to list of fields
        if type(fields) is str:
            fields = [fields]
            # also need to form sweep_guesses to right shape 
        n_fields = len(fields)
        if bounds:
            bounds = [ b for b in bounds for i in range(n_exps)]
        # Define some short names for commonly used properties.
        up = self._updater
        m  = up.model
        
        if opt_options is None:
            opt_options = {}
        # Generate a new guess or use a guess provided, depending on the
        # type of the guess argument.
        if isinstance(guess, Heuristic):
            raise NotImplementedError("Not yet implemented.")
        elif callable(guess):
            # Generate a new guess by calling the guess function provided.
            ep = guess(
                idx_exp=len(up.data_record),
                mean=up.est_mean(),
                cov=up.est_covariance_mtx()
            )
        else:
            # Make a copy of the guess that we can manipulate, but otherwise
            # use it as-is.

            #handle case where multiple guesses are provided for sweep guesses
            # I feel that this whole class could be reworked with a better design
            # -Thomas 

       
            ep = np.copy(guess).reshape(-1)
        
        self.loss = (0,0,10000000)
        # Define an objective function that wraps a vector of scalars into
        # an appropriate record array.
        if (cost_mult==False):
            def objective_function(x):
                """
                Used internally by design_expparams_field.
                If you see this, something probably went wrong.
                """
                for i in range(n_exps):       
                    for j,f in enumerate(fields): 
                        ep[i:i+1][f] = x[i*n_fields+j]
                
                cost = np.sum(self._utility_func(ep) + cost_scale_k * m.experiment_cost(ep))
                return cost
        else:
            def objective_function(x):
                """
                Used internally by design_expparams_field.
                If you see this, something probably went wrong.
                """

                old_ep = np.copy(ep)
                for i in range(n_exps):       
                    for j,f in enumerate(fields): 
                        ep[i:i+1][f] = x[i*n_fields+j]

                cost = np.sum(self._utility_func(ep)* m.experiment_cost(ep)**cost_scale_k)
                return cost
        
            
        
        # Allocate a variable to hold the local optimum value found.
        # This way, if an optimization algorithm doesn't support returning
        # the value as well as the location, we can find it manually.
        f_opt = None
            
        # Here's the core, where we break out and call the various optimization
        # routines provided by SciPy.
        if self._opt_algo == OptimizationAlgorithms.NULL:
            # This optimization algorithm does nothing locally, but only
            # exists to leverage the store_guess functionality below.
            
            x_opt = [ guess[i][f] for f in fields for i in range(n_exps)]
            x_opt = np.array(x_opt)
            
        elif self._opt_algo == OptimizationAlgorithms.CG:
            # Prepare any additional options.
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
                
            # Actually call fmin_cg, gathering all outputs we can.
            x_opt, f_opt, func_calls, grad_calls, warnflag = opt.fmin_cg(
                objective_function, [ guess[i][f] for f in fields for i in range(n_exps)],
                disp=disp, full_output=True, **opt_options
            )
            
        elif self._opt_algo == OptimizationAlgorithms.NCG:
            # Prepare any additional options.
        
            if maxfun is not None:
                opt_options['maxfun'] = maxfun
            if grad_h is not None:
                opt_options['epsilon'] = grad_h
                
            # Actually call fmin_tnc, gathering all outputs we can.
            # We use fmin_tnc in preference to fmin_ncg, as they implement the
            # same algorithm, but fmin_tnc seems better behaved with respect
            # to very flat gradient regions, due to respecting maxfun.
            # By contrast, fmin_ncg can get stuck in an infinite loop in
            # versions of SciPy < 0.11.
            #
            # Note that in some versions of SciPy, there was a bug in
            # fmin_ncg and fmin_tnc that can propagate outward if the gradient
            # is too flat. We catch it here and return the initial guess in that
            # case, since by hypothesis, it's too flat to make much difference
            # anyway.
            try:
                x_opt, f_opt, func_calls, grad_calls, h_calls, warnflag = opt.fmin_tnc(
                    objective_function, [ guess[i][f] for f in fields for i in range(n_exps)],
                    fprime=None, bounds=None, approx_grad=True,
                    disp=disp, full_output=True, **opt_options
                )
            except TypeError:
                warnings.warn(
                    "Gradient function too flat for NCG.",
                    RuntimeWarning)
                x_opt = np.array([ guess[i][f] for f in fields for i in range(n_exps)])
                f_opt = None
                
        elif self._opt_algo == OptimizationAlgorithms.NELDER_MEAD:
    
            if maxfun is not None:
                opt_options['maxfun'] = maxfun
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
                
            x_opt, f_opt, iters, func_calls, warnflag = opt.fmin(
                objective_function, [ guess[i][f] for f in fields for i in range(n_exps)],
                disp=disp, full_output=True, **opt_options
            )
        
        elif self._opt_algo == OptimizationAlgorithms.L_BFGS_B:
            if maxfun is not None:
                opt_options['maxfun'] = maxfun
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
            if grad_h is not None:
                opt_options['epsilon'] = grad_h
            if bounds is not None:
                opt_options['bounds'] = bounds

            x_opt,f_opt,d = opt.fmin_l_bfgs_b(objective_function,
                [ guess[i][f] for f in fields for i in range(n_exps)],approx_grad=True,
                    disp=disp,**opt_options)

            if disp > 0 :
                print ("grad:{0}   function calls:{1}    iterations:{2}".format(
                    d['grad'],d['funcalls'],d['nit']))

        elif self._opt_algo == OptimizationAlgorithms.RANDOM_GUESSES:
            if n_samples is None:
                raise ValueError('parameter n_samples must be set for '
                    'optimization algorithm RANDOM_GUESSES')
            
            b = np.array(bounds)
            exps = np.random.uniform(np.tile(b[:,0],(n_samples,1)),np.tile(b[:,1],(n_samples,1)))
            risks = [objective_function(e)for e in exps]
            x_opt,f_opt = exps[np.argmin(risks)], np.amin(risks)   
        
        elif self._opt_algo == OptimizationAlgorithms.LINEAR_SWEEP:
            if n_samples is None:
                raise ValueError('parameter n_samples must be set for '
                    'optimization algorithm LINEAR_SWEEP')
            self._updater.stds = []
            b = np.array(bounds)
            lins = []
            n_bounds = len(bounds)
            if n_bounds>1:
                n_lin = np.ceil(np.power(float(n_samples),1./n_bounds))
            else:
                n_lin = n_samples
                
            for i in range(n_bounds):
                lins.append(np.linspace(bounds[i][0],bounds[i][1],n_lin))
            
            exps = np.array(np.meshgrid(*lins,indexing='ij')).T.reshape(-1,n_bounds)[:n_samples]
            risks = [objective_function(e)for e in exps]
            
            x_opt,f_opt = exps[np.argmin(risks)], np.amin(risks)  
            
        elif self._opt_algo == OptimizationAlgorithms.BASIN_HOPPING:
            if not isinstance(niter,int):
                raise ValueError('For BASIN_HOPPING niter must be set')
            elif niter<=0:
                raise ValueError('niter must be an integer >0')
            
            minimizer_kwargs = {'method':"L-BFGS-B",'options':{}}
            
            if bounds: 
                minimizer_kwargs['bounds'] = bounds
                b = np.array(bounds)
                take_step = RandomDisplacementBounds(b[:,0],b[:,1])
                opt_options['take_step'] = take_step
            if 'epsilon' in opt_options:
                minimizer_kwargs['options']['eps'] = opt_options.pop('epsilon',1e-8)

            res = opt.basinhopping(objective_function,[ guess[i][f] for f in fields for i in range(n_exps)],minimizer_kwargs=minimizer_kwargs,
                                   niter=niter,disp=disp,**opt_options)
          
            x_opt = res.x 
            f_opt = res.fun


      
        


        # Optionally compare the result to previous guesses.           
        if store_guess:
            # Possibly compute the objective function value at the local optimum
            # if we don't already know it.
            if f_opt is None:
                guess_qual = objective_function(*x_opt)
            
            # Compare to the known best cost so far.
            if self.__best_cost is None or (self.__best_cost > f_opt):
                # No known best yet, or we're better than the previous best,
                # so record this guess.
                for i in range(n_exps):       
                    for j,f in enumerate(fields): 
                        ep[i:i+1][f] = x_opt[i*n_fields+j]
                
                self.__best_cost = f_opt
                self.__best_ep = ep
            else:
                ep = self.__best_ep # Guess is bad, return current best guess
        else:
            # We aren't using guess recording, so just pack the local optima
            # into ep for returning.
            for i in range(n_exps):       
                    for j,f in enumerate(fields): 
                        ep[i:i+1][f] = x_opt[i*n_fields+j]
        # In any case, return the optimized gu
        
        self._num_calls +=1
        return ep

class Bounds(object):

    def __init__(self,xmin,xmax,dtype=np.float32):
        self.xmax = np.array(xmax,dtype=dtype)
        self.xmin = np.array(xmin,dtype=dtype)

    def __call__(self,**kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        bound = tmax and tmin
        return bound
        
class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        while True:
            # this could be done in a much more clever way, but it will work for example purposes
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
                break
        return xnew