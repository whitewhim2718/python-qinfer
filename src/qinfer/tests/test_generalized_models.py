#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# generalized_models_test.py: Checks that the Generalized models works properly.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
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
from __future__ import absolute_import
## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer.abstract_model import (
    FiniteOutcomeModel)
from qinfer import (GaussianModel,BasicGaussianModel,PoissonModel,BasicPoissonModel,
                    UniformDistribution)

from qinfer.smc import SMCUpdater,SMCUpdaterBCRB

class ExponentialGaussianModel(GaussianModel):
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


class ExponentialPoissonModel(PoissonModel):
    """
    The basic Gaussian model consisting of a single model parameter :math:`\mu`,
    and no experimental parameters.
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

    


class TestGaussianModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([79,3],dtype=np.float)
    ONLINE_SIGMA = 0.2
    TEST_EXPPARAMS = np.linspace(1.,500.,10000,dtype=np.float)
    PRIOR_NO_SIGMA_PARAM = UniformDistribution([[0,100]])
    PRIOR_SIGMA_PARAM = UniformDistribution([[0,100],[0,10]])
    N_PARTICLES = 10000
    N_BIM = 1000
    N_ONLINE = 50  
    N_OUTCOME_SAMPLES = 500
    TEST_EXPPARAMS_RISK = np.linspace(1.,500.,N_ONLINE,dtype=np.float)
    N_GUESSES = 50
    MAX_EXPPARAM = 500.,
    TEST_TARGET_COV_NO_SIGMA_PARAM = np.array([[0.1]])
    TEST_TARGET_COV_SIGMA_PARAM = np.array([[0.1,0.1],[0.1,0.1]])


    def setUp(self):

        super(TestGaussianModel,self).setUp()
        sigma = TestGaussianModel.MODELPARAMS[1]
        
        self.gaussian_model_no_sigma_param = BasicGaussianModel(sigma=sigma,
                                            num_outcome_samples=TestGaussianModel.N_OUTCOME_SAMPLES)
        self.gaussian_model_sigma_param = BasicGaussianModel(num_outcome_samples=TestGaussianModel.N_OUTCOME_SAMPLES)
        self.exponential_gaussian_model = ExponentialGaussianModel(sigma=TestGaussianModel.ONLINE_SIGMA,
                                            num_outcome_samples=TestGaussianModel.N_OUTCOME_SAMPLES)

        self.expparams = TestGaussianModel.TEST_EXPPARAMS.reshape(-1,1)
        self.expparams_risk = TestGaussianModel.TEST_EXPPARAMS_RISK.reshape(-1,1)
        self.outcomes_no_sigma_param = self.gaussian_model_no_sigma_param.simulate_experiment(TestGaussianModel.MODELPARAMS[:1],
                TestGaussianModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)
        self.outcomes_sigma_param = self.gaussian_model_sigma_param.simulate_experiment(TestGaussianModel.MODELPARAMS,
                TestGaussianModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.outcomes_exponential = self.exponential_gaussian_model.simulate_experiment(TestGaussianModel.MODELPARAMS[:1],
                TestGaussianModel.TEST_EXPPARAMS_RISK.astype(self.exponential_gaussian_model.expparams_dtype),
                repeat=1 ).reshape(-1,1)


        self.updater_no_sigma_param = SMCUpdater(self.gaussian_model_no_sigma_param,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_NO_SIGMA_PARAM)

        self.updater_sigma_param = SMCUpdater(self.gaussian_model_sigma_param,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_SIGMA_PARAM)

        self.updater_bayes_no_sigma_param = SMCUpdaterBCRB(self.gaussian_model_no_sigma_param,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_NO_SIGMA_PARAM,adaptive=True)

        self.updater_bayes_sigma_param = SMCUpdaterBCRB(self.gaussian_model_sigma_param,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_SIGMA_PARAM,adaptive=True)

        self.exponential_updater_one_guess = SMCUpdater(self.exponential_gaussian_model,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_NO_SIGMA_PARAM)

        self.exponential_updater_many_guess = SMCUpdater(self.exponential_gaussian_model,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_NO_SIGMA_PARAM)

        self.exponential_updater_sweep = SMCUpdater(self.exponential_gaussian_model,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_NO_SIGMA_PARAM)


    def test_gaussian_model_fitting(self):

        self.updater_no_sigma_param.batch_update(self.outcomes_no_sigma_param,self.expparams,5)
        self.updater_sigma_param.batch_update(self.outcomes_sigma_param,self.expparams,5)

        assert_almost_equal(self.updater_no_sigma_param.est_mean(),TestGaussianModel.MODELPARAMS[:1],0)
        assert_almost_equal(self.updater_sigma_param.est_mean(),TestGaussianModel.MODELPARAMS,0)

        assert_array_less(self.updater_no_sigma_param.est_covariance_mtx(),TestGaussianModel.TEST_TARGET_COV_NO_SIGMA_PARAM)
        assert_array_less(self.updater_sigma_param.est_covariance_mtx(),TestGaussianModel.TEST_TARGET_COV_SIGMA_PARAM)

    def test_gaussian_bim(self):
        """
        Checks that the fitters converge on true value on simple precession_model. Is a stochastic
        test but I ran 100 times and there were no fails, with these parameters.
        """
        bim_currents_no_sigma_param = []
        bim_adaptives_no_sigma_param = []

        bim_currents_sigma_param = []
        bim_adaptives_sigma_param = []

        #track bims throughout experiments
        for i in range(TestGaussianModel.N_BIM):         
            self.updater_bayes_no_sigma_param.update(self.outcomes_no_sigma_param[i],self.expparams[i])
            self.updater_bayes_sigma_param.update(self.outcomes_sigma_param[i],self.expparams[i])

            bim_currents_no_sigma_param.append(self.updater_bayes_no_sigma_param.current_bim)
            bim_adaptives_no_sigma_param.append(self.updater_bayes_no_sigma_param.adaptive_bim)
            bim_currents_sigma_param.append(self.updater_bayes_sigma_param.current_bim)
            bim_adaptives_sigma_param.append(self.updater_bayes_sigma_param.adaptive_bim)

        bim_currents_no_sigma_param = np.array(bim_currents_no_sigma_param)
        bim_adaptives_no_sigma_param = np.array(bim_adaptives_no_sigma_param)
        bim_currents_sigma_param = np.array(bim_currents_sigma_param)
        bim_adaptives_sigma_param = np.array(bim_adaptives_sigma_param)


        #verify that BCRB is approximately reached 
        assert_almost_equal(self.updater_bayes_no_sigma_param.est_covariance_mtx(),
            np.linalg.inv(self.updater_bayes_no_sigma_param.current_bim),1)
        assert_almost_equal(self.updater_bayes_no_sigma_param.est_covariance_mtx(),
            np.linalg.inv(self.updater_bayes_no_sigma_param.adaptive_bim),1)

        assert_almost_equal(self.updater_bayes_sigma_param.est_covariance_mtx(),
            np.linalg.inv(self.updater_bayes_sigma_param.current_bim),1)
        assert_almost_equal(self.updater_bayes_sigma_param.est_covariance_mtx(),
            np.linalg.inv(self.updater_bayes_sigma_param.adaptive_bim),1)


    def test_bayes_risk(self):
        opt_exps_one_guess = []
        opt_exps_risk_many_guess = []
        opt_exps_ig_many_guess = []

        # classic sweep to check against
   
        self.exponential_updater_sweep.batch_update(self.outcomes_exponential,self.expparams_risk.astype(
            self.exponential_gaussian_model.expparams_dtype),5)

        for i in range(TestGaussianModel.N_ONLINE):

            guesses = np.random.uniform(low=0.,high=TestGaussianModel.MAX_EXPPARAM,
                size=TestGaussianModel.N_GUESSES).reshape(-1,1).astype(
                        self.exponential_gaussian_model.expparams_dtype)
            
            risks = []
            igs = []
            for i,g in enumerate(guesses):
                risks.append(self.exponential_updater_many_guess.bayes_risk(guesses[i]))
                #igs.append(self.exponential_updater_many_guess.expected_information_gain(guesses[i]))

            risks = np.array(risks)
            #igs = np.array(igs)
            one_guess_exp = guesses[0]
            many_guess_exp = guesses[np.argmin(risks)]
            #many_guess_exp_ig = guesses[np.argmin(igs)]

            opt_exps_one_guess.append(one_guess_exp)
            opt_exps_risk_many_guess.append(many_guess_exp)
            #opt_exps_ig_many_guess.append(many_guess_exp_ig)

            outcome_one_guess = self.exponential_gaussian_model.simulate_experiment(TestGaussianModel.MODELPARAMS[:1],
                one_guess_exp,repeat=1 )[0]
            outcome_many_guess = self.exponential_gaussian_model.simulate_experiment(TestGaussianModel.MODELPARAMS[:1],
                many_guess_exp,repeat=1 )[0]

            self.exponential_updater_one_guess.update(outcome_one_guess,one_guess_exp)
            self.exponential_updater_many_guess.update(outcome_many_guess,many_guess_exp)
    
        assert_almost_equal(self.exponential_updater_many_guess.est_mean(),TestGaussianModel.MODELPARAMS[:1],-1)
        assert_almost_equal(self.exponential_updater_sweep.est_mean(),TestGaussianModel.MODELPARAMS[:1],-1)
        assert_array_less(self.exponential_updater_many_guess.est_covariance_mtx(),
                            self.exponential_updater_one_guess.est_covariance_mtx())
        assert_array_less(self.exponential_updater_many_guess.est_covariance_mtx(),
                            self.exponential_updater_sweep.est_covariance_mtx())



class TestPoissonModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([79.,])
    MODELPARAMS_RISK = np.array([79.,])
    TEST_EXPPARAMS = np.arange(50000,dtype=np.float)
    PRIOR = UniformDistribution([[0.,200.]])
    N_PARTICLES = 10000
    N_ONLINE = 50
    N_GUESSES = 50
    N_OUTCOME_SAMPLES = 500
    MAX_EXPPARAM = 500.
    TEST_EXPPARAMS_RISK = np.linspace(1.,MAX_EXPPARAM,N_ONLINE,dtype=np.float)
    N_BIM = 1000
    TEST_TARGET_COV = np.array([[0.1]])

    def setUp(self):

        super(TestPoissonModel,self).setUp()
        self.poisson_model = BasicPoissonModel(num_outcome_samples=TestPoissonModel.N_OUTCOME_SAMPLES)
        self.exponential_poisson_model = ExponentialPoissonModel(num_outcome_samples=TestPoissonModel.N_OUTCOME_SAMPLES)


        self.expparams = TestPoissonModel.TEST_EXPPARAMS.reshape(-1,1)
        self.expparams_risk = TestPoissonModel.TEST_EXPPARAMS_RISK.reshape(-1,1)

        self.outcomes = self.poisson_model.simulate_experiment(TestPoissonModel.MODELPARAMS,
                TestPoissonModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.outcomes_exponential = self.exponential_poisson_model.simulate_experiment(TestPoissonModel.MODELPARAMS_RISK,
                TestPoissonModel.TEST_EXPPARAMS_RISK.astype(self.exponential_poisson_model.expparams_dtype),
                repeat=1 ).reshape(-1,1)


        self.updater = SMCUpdater(self.poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)

        self.updater_bayes = SMCUpdaterBCRB(self.poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR,adaptive=True)

        self.exponential_updater_one_guess = SMCUpdater(self.exponential_poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)

        self.exponential_updater_many_guess = SMCUpdater(self.exponential_poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)

        self.exponential_updater_sweep = SMCUpdater(self.exponential_poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)
        
    def test_poisson_model_fitting(self):

        self.updater.batch_update(self.outcomes,self.expparams,5)


        assert_almost_equal(self.updater.est_mean(),TestPoissonModel.MODELPARAMS,2)
        assert_array_less(self.updater.est_covariance_mtx(),TestPoissonModel.TEST_TARGET_COV)

    def test_poisson_bim(self):
        """
        Checks that the fitters converge on true value on simple precession_model. Is a stochastic
        test but I ran 100 times and there were no fails, with these parameters.
        """
        bim_currents = []
        bim_adaptives = []

        #track bims throughout experiments
        for i in range(TestPoissonModel.N_BIM):         
            self.updater_bayes.update(self.outcomes[i],self.expparams[i])

            bim_currents.append(self.updater_bayes.current_bim)
            bim_adaptives.append(self.updater_bayes.adaptive_bim)

        bim_currents = np.array(bim_currents)
        bim_adaptives = np.array(bim_adaptives)


        #verify that BCRB is approximately reached 
        assert_almost_equal(self.updater_bayes.est_covariance_mtx(),np.linalg.inv(self.updater_bayes.current_bim),1)
        assert_almost_equal(self.updater_bayes.est_covariance_mtx(),np.linalg.inv(self.updater_bayes.adaptive_bim),1)


    def test_bayes_risk(self):
        opt_exps_one_guess = []
        opt_exps_risk_many_guess = []
        opt_exps_ig_many_guess = []

        # classic sweep to check against
   
        self.exponential_updater_sweep.batch_update(self.outcomes_exponential,self.expparams_risk.astype(
            self.exponential_poisson_model.expparams_dtype),5)

        for i in range(TestPoissonModel.N_ONLINE):

            guesses = np.random.uniform(low=0.,high=TestPoissonModel.MAX_EXPPARAM,
                size=TestPoissonModel.N_GUESSES).reshape(-1,1).astype(
                        self.exponential_poisson_model.expparams_dtype)
            
            risks = []
            igs = []
            for i,g in enumerate(guesses):
                risks.append(self.exponential_updater_many_guess.bayes_risk(guesses[i]))
                #igs.append(self.exponential_updater_many_guess.expected_information_gain(guesses[i]))
           
            risks = np.array(risks)
            #igs = np.array(igs)
            one_guess_exp = guesses[0]
            many_guess_exp = guesses[np.argmin(risks)]
            #many_guess_exp_ig = guesses[np.argmin(igs)]

            opt_exps_one_guess.append(one_guess_exp)
            opt_exps_risk_many_guess.append(many_guess_exp)
            #opt_exps_ig_many_guess.append(many_guess_exp_ig)

            outcome_one_guess = self.exponential_poisson_model.simulate_experiment(TestPoissonModel.MODELPARAMS_RISK,
                one_guess_exp,repeat=1 )[0]
            outcome_many_guess = self.exponential_poisson_model.simulate_experiment(TestPoissonModel.MODELPARAMS_RISK,
                many_guess_exp,repeat=1 )[0]

            self.exponential_updater_one_guess.update(outcome_one_guess,one_guess_exp)
            self.exponential_updater_many_guess.update(outcome_many_guess,many_guess_exp)
        
       
        assert_almost_equal(self.exponential_updater_many_guess.est_mean(),TestPoissonModel.MODELPARAMS_RISK,-1)
        assert_almost_equal(self.exponential_updater_sweep.est_mean(),TestPoissonModel.MODELPARAMS_RISK,-1)
        assert_array_less(self.exponential_updater_many_guess.est_covariance_mtx(),
                            self.exponential_updater_one_guess.est_covariance_mtx())
        assert_array_less(self.exponential_updater_many_guess.est_covariance_mtx(),
                            self.exponential_updater_sweep.est_covariance_mtx())






