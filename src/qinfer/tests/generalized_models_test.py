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
    Model)
from qinfer import GaussianModel,PoissonModel,MultinomialModel

from qinfer.smc import SMCUpdater



class TestPoissonModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([1,])
    TEST_EXPPARAMS = np.linspace(1.,10.,100,dtype=np.float)
    PRIOR = UniformDistribution([[0,2]])
    N_PARTICLES = 10000

    TEST_TARGET_COV = np.array([[0.01]])

    def setUp(self):

        super(TestPoissonModel,self).setUp()
        self.poisson_model = PoissonModel()
        self.expparams = TestPoissonModel.TEST_EXPPARAMS.reshape(-1,1)
        self.outcomes = self.poisson_model.simulate_experiment(TestPoissonModel.MODELPARAMS,
                TestPoissonModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.updater = TestPoissonModel(self.poisson_model,
                TestPoissonModel.N_PARTICLES,TestPoissonModel.PRIOR)
       

class TestMultinomialModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([1,])
    TEST_EXPPARAMS = np.linspace(1.,10.,100,dtype=np.float)
    PRIOR = UniformDistribution([[0,2]])
    N_PARTICLES = 10000

    TEST_TARGET_COV = np.array([[0.01]])

    def setUp(self):

        super(TestMultinomialModel,self).setUp()
        self.multinomial_model = MultinomialModel()
        self.expparams = TestMultinomialModel.TEST_EXPPARAMS.reshape(-1,1)
        self.outcomes = self.multinomial_model.simulate_experiment(TestMultinomialModel.MODELPARAMS,
                TestMultinomialModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.updater = TestMultinomialModel(self.multinomial_model,
                TestMultinomialModel.N_PARTICLES,TestMultinomialModel.PRIOR)



class TestGaussianModel(DerandomizedTestCase):
    # True model parameter for test
    MODELPARAMS = np.array([1,])
    TEST_EXPPARAMS = np.linspace(1.,10.,100,dtype=np.float)
    PRIOR = UniformDistribution([[0,2]])
    N_PARTICLES = 10000

    TEST_TARGET_COV = np.array([[0.01]])

    def setUp(self):

        super(TestGaussianModel,self).setUp()
        self.gaussian_model = GaussianModel()
        self.expparams = TestGaussianModel.TEST_EXPPARAMS.reshape(-1,1)
        self.outcomes = self.gaussian_model.simulate_experiment(TestGaussianModel.MODELPARAMS,
                TestGaussianModel.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

        self.updater = TestGaussianModel(self.gaussian_model,
                TestGaussianModel.N_PARTICLES,TestGaussianModel.PRIOR)