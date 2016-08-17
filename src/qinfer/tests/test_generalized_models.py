#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_generalized_models.py: Checks that models with generalized outcomes work.
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
                    ExponentialGaussianModel,ExponentialPoissonModel,
                     UniformDistribution)

from qinfer.smc import SMCUpdater,SMCUpdaterBCRB

    
class TestGaussianModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([79,3],dtype=np.float)
    ONLINE_SIGMA = 0.2
    TEST_EXPPARAMS = np.linspace(1.,1000.,10000,dtype=np.float)
    PRIOR_NO_SIGMA_PARAM = UniformDistribution([[0,100]])
    PRIOR_SIGMA_PARAM = UniformDistribution([[0,100],[0,10]])
    N_PARTICLES = 10000
    N_BIM = 1000
    N_ONLINE = 25  
    N_GUESSES = 25
    N_OUTCOME_SAMPLES = 250
    TEST_EXPPARAMS_RISK = np.linspace(1.,500.,N_ONLINE,dtype=np.float)
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



class TestPoissonModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([79.,])
    MODELPARAMS_RISK = np.array([79.,])
    TEST_EXPPARAMS = np.arange(0, 5000,50000,dtype=np.float)
    PRIOR = UniformDistribution([[0.,200.]])
    N_PARTICLES = 10000
    N_ONLINE = 25
    N_GUESSES = 25
    N_OUTCOME_SAMPLES = 250
    MAX_EXPPARAM = 500.
    TEST_EXPPARAMS_RISK = np.linspace(1.,MAX_EXPPARAM,N_ONLINE,dtype=np.float)
    N_BIM = 1000
    TEST_TARGET_COV = np.array([[0.1]])

    def setUp(self):

        super(TestPoissonModel,self).setUp()
        self.poisson_model = BasicPoissonModel(num_outcome_samples=TestPoissonModel.N_OUTCOME_SAMPLES)
        self.exponential_poisson_model = ExponentialPoissonModel(num_outcome_samples=TestPoissonModel.N_OUTCOME_SAMPLES)


        self.expparams = TestPoissonModel.TEST_EXPPARAMS.reshape(-1,1).astype(self.poisson_model.expparams_dtype)
        self.expparams_risk = TestPoissonModel.TEST_EXPPARAMS_RISK.reshape(-1,1).astype(self.poisson_model.expparams_dtype)

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






