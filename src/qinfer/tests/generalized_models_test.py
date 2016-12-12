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
from qinfer import BasicGaussianModel,BasicPoissonModel,MultinomialModel,UniformDistribution

from qinfer.smc import SMCUpdater



class TestGaussianModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([79,3])
    TEST_EXPPARAMS = np.linspace(1.,10.,10000,dtype=np.float)
    PRIOR_NO_SIGMA_PARAM = UniformDistribution([[0,100]])
    PRIOR_SIGMA_PARAM = UniformDistribution([[0,100],[0,10]])
    N_PARTICLES = 10000

    TEST_TARGET_COV_NO_SIGMA_PARAM = np.array([[0.1]])
    TEST_TARGET_COV_SIGMA_PARAM = np.array([[0.1,0.1],[0.1,0.1]])

    def setUp(self):

        super(TestGaussianModel,self).setUp()
        sigma = TestGaussianModel.MODELPARAMS[1]
        self.gaussian_model_no_sigma_param = BasicGaussianModel(sigma=sigma)
        self.gaussian_model_sigma_param = BasicGaussianModel()
        self.expparams = TestGaussianModel.TEST_EXPPARAMS.reshape(-1,1)
        self.outcomes_no_sigma_param = self.gaussian_model_no_sigma_param.simulate_experiment(TestGaussianModel.MODELPARAMS[:1],
                TestGaussianModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)
        self.outcomes_sigma_param = self.gaussian_model_sigma_param.simulate_experiment(TestGaussianModel.MODELPARAMS,
                TestGaussianModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.updater_no_sigma_param = SMCUpdater(self.gaussian_model_no_sigma_param,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_NO_SIGMA_PARAM)

        self.updater_sigma_param = SMCUpdater(self.gaussian_model_sigma_param,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR_SIGMA_PARAM)


    def test_gaussian_model_fitting(self):

        self.updater_no_sigma_param.batch_update(self.outcomes_no_sigma_param,self.expparams,5)
        self.updater_sigma_param.batch_update(self.outcomes_sigma_param,self.expparams,5)

        assert_almost_equal(self.updater_no_sigma_param.est_mean(),TestGaussianModel.MODELPARAMS[:1],0)
        assert_almost_equal(self.updater_sigma_param.est_mean(),TestGaussianModel.MODELPARAMS,0)

        assert_array_less(self.updater_no_sigma_param.est_covariance_mtx(),TestGaussianModel.TEST_TARGET_COV_NO_SIGMA_PARAM)
        assert_array_less(self.updater_sigma_param.est_covariance_mtx(),TestGaussianModel.TEST_TARGET_COV_SIGMA_PARAM)


class TestPoissonModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([79.,])
    TEST_EXPPARAMS = np.arange(50000,dtype=np.float)
    PRIOR = UniformDistribution([[0.,100.]])
    N_PARTICLES = 10000
    N_ONLINE = 50
    TEST_TARGET_COV = np.array([[0.1]])

    def setUp(self):

        super(TestPoissonModel,self).setUp()
        self.poisson_model = BasicPoissonModel()
        self.expparams = TestPoissonModel.TEST_EXPPARAMS.reshape(-1,1)
        self.outcomes = self.poisson_model.simulate_experiment(TestPoissonModel.MODELPARAMS,
                TestPoissonModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.updater = SMCUpdater(self.poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)

        self.updater_online = SMCUpdater(self.poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)
    

    def test_poisson_model_fitting(self):

        self.updater.batch_update(self.outcomes,self.expparams,5)


        assert_almost_equal(self.updater.est_mean(),TestPoissonModel.MODELPARAMS,2)
        assert_array_less(self.updater.est_covariance_mtx(),TestPoissonModel.TEST_TARGET_COV)

    #def test_bayes_risk_performance(self):
    #
    #    for i in N_ONLINE:
    #        outcomes = self.poisson_model.simulate_experiment








