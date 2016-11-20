#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_metrics.py: Tests various metrics like risk and information gain.
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
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less,assert_approx_equal

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer import (BasicPoissonModel,BasicGaussianModel,BinomialModel,CoinModel,GammaDistribution,BetaDistribution,NormalDistribution)

from qinfer.smc import SMCUpdater,SMCUpdaterBCRB

class TestBayesRisk(DerandomizedTestCase):
    # Test the implementation of numerical Bayes Risk by comparing to 
    # numbers which were derived by doing analytic/numeric
    # integrals of simple models (binomialm, poisson, and gaussian) in 
    # Mathematica. This test trusts that these calculations
    # were done correctly.

    ALPHA = 1.
    BETA = 3.
    MU = 1.0
    VAR = 1.0
    VAR_LIKELIHOOD = 1.0
    PRIOR_BETA = BetaDistribution(alpha=ALPHA, beta=BETA)
    PRIOR_GAMMA = GammaDistribution(alpha=ALPHA, beta=BETA)
    PRIOR_NORMAL = NormalDistribution(mean=MU,var=VAR)
    N_PARTICLES = 10000
    N_OUTCOME_SAMPLES = 1000
    TAU_EXPPARAMS = np.arange(1, 11, dtype=int)
    NMEAS_EXPPARAMS = np.arange(1, 11, dtype=int)
    
    def setUp(self):

        super(TestBayesRisk,self).setUp()
        
        # Set up relevant models.
        self.poisson_model = BasicPoissonModel(num_outcome_samples=TestBayesRisk.N_OUTCOME_SAMPLES)
        self.gaussian_model = BasicGaussianModel(var=TestBayesRisk.VAR_LIKELIHOOD,
            num_outcome_samples=TestBayesRisk.N_OUTCOME_SAMPLES)
        self.coin_model = CoinModel()

        self.binomial_model = BinomialModel(self.coin_model)

        # Set up updaters for these models using particle approximations 
        # of conjugate priors
        self.updater_poisson = SMCUpdater(self.poisson_model,
                TestBayesRisk.N_PARTICLES,TestBayesRisk.PRIOR_GAMMA)
        self.updater_binomial = SMCUpdater(self.binomial_model,
                TestBayesRisk.N_PARTICLES,TestBayesRisk.PRIOR_BETA)
        self.updater_gaussian = SMCUpdater(self.gaussian_model,
                TestBayesRisk.N_PARTICLES,TestBayesRisk.PRIOR_NORMAL)

    def test_finite_outcomes_risk(self):
        # The binomial model has a finite number of outcomes. Test the 
        # risk calculation in this case.

        expparams = self.NMEAS_EXPPARAMS.astype(self.binomial_model.expparams_dtype)

        # estimate the risk
        est_risk = self.updater_binomial.bayes_risk(expparams)

        # compute exact risk
        a, b = TestBayesRisk.ALPHA, TestBayesRisk.BETA
        exact_risk = a * b / ((a + b) * (a + b + 1) * (a + b + expparams['n_meas']))

        # see if they roughly match
        assert_almost_equal(est_risk, exact_risk, decimal=1)

    def test_infinite_outcomes_risk(self):
        # The poisson model has a (countably) infinite number of outcomes. Test the 
        # risk calculation in this case.

        expparams = self.TAU_EXPPARAMS.astype(self.poisson_model.expparams_dtype)

        # estimate the risk
        est_risk = self.updater_poisson.bayes_risk(expparams)

        # compute exact risk
        a, b = TestBayesRisk.ALPHA, TestBayesRisk.BETA
        exact_risk = a / (b * (b + expparams['tau']))

        # see if they roughly match
        assert_almost_equal(est_risk, exact_risk, decimal=1)

    def test_continuous_outcomes_risk(self):
        # The gaussian model has a (uncountably) infinite number of outcomes. Test the
        # risk calculation in this case. 

        expparams = self.TAU_EXPPARAMS.astype(self.gaussian_model.expparams_dtype)

        # estimate the risk
        est_risk = self.updater_gaussian.bayes_risk(expparams,use_cached_samples=True)

        #compute the exact risk 
        mu, var, var_lik = TestBayesRisk.MU, TestBayesRisk.VAR, \
                            TestBayesRisk.VAR_LIKELIHOOD

        exact_risk = var*var_lik/((var+var_lik*expparams['tau']**2))

        assert_almost_equal(est_risk, exact_risk, decimal=1)



class TestInformationGain(DerandomizedTestCase):
    # Test the implementation of numerical information gain by comparing to 
    # numbers which were derived by doing analytic/numeric
    # integrals of simple models (binomialm, poisson, and gaussian) in 
    # Mathematica. This test trusts that these calculations
    # were done correctly.

    ALPHA = 1
    BETA = 3
    MU = 1.0
    VAR = 10.0
    VAR_LIKELIHOOD = 0.2
    PRIOR_BETA = BetaDistribution(alpha=ALPHA, beta=BETA)
    PRIOR_GAMMA = GammaDistribution(alpha=ALPHA, beta=BETA)
    PRIOR_NORMAL = NormalDistribution(mean=MU,var=VAR)
    N_PARTICLES = 10000
    N_OUTCOME_SAMPLES = 1000
    # Calculated in Mathematica, IG for the binomial model and the given expparams
    NMEAS_EXPPARAMS = np.arange(1, 11, dtype=int)
    BINOM_IG = np.array([0.104002,0.189223,0.261496,0.324283,0.379815,0.429613,0.474764,0.516069,0.554138,0.589446])
    
    # Calculated in Mathematica, IG for the poisson model and the given expparams
    TAU_EXPPARAMS = np.arange(1, 11, dtype=int)
    POISSON_IG = np.array([0.123097,0.220502,0.301245,0.370271,0.430595,0.484192,0.532429,0.576292,0.616517,0.653667])

    def setUp(self):

        super(TestInformationGain,self).setUp()
        
        # Set up relevant models.
        self.poisson_model = BasicPoissonModel(num_outcome_samples=TestInformationGain.N_OUTCOME_SAMPLES)
        self.coin_model = CoinModel()
        self.binomial_model = BinomialModel(self.coin_model)
        self.gaussian_model = BasicGaussianModel(var=TestInformationGain.VAR_LIKELIHOOD,
            num_outcome_samples=TestInformationGain.N_OUTCOME_SAMPLES)

        # Set up updaters for these models using particle approximations 
        # of conjugate priors
        self.updater_poisson = SMCUpdater(self.poisson_model,
                TestInformationGain.N_PARTICLES,TestInformationGain.PRIOR_GAMMA)
        self.updater_binomial = SMCUpdater(self.binomial_model,
                TestInformationGain.N_PARTICLES,TestInformationGain.PRIOR_BETA)
        self.updater_gaussian = SMCUpdater(self.gaussian_model,
                TestInformationGain.N_PARTICLES,TestInformationGain.PRIOR_NORMAL)


    def test_finite_outcomes_ig(self):
        # The binomial model has a finite number of outcomes. Test the 
        # ig calculation in this case.

        expparams = self.NMEAS_EXPPARAMS.astype(self.binomial_model.expparams_dtype)

        # estimate the information gain
        est_ig = self.updater_binomial.expected_information_gain(expparams)

        # see if they roughly match
        assert_almost_equal(est_ig, TestInformationGain.BINOM_IG, decimal=1)

    def test_infinite_outcomes_ig(self):
        # The poisson model has a (countably) infinite number of outcomes. Test the 
        # ig calculation in this case.

        expparams = self.TAU_EXPPARAMS.astype(self.poisson_model.expparams_dtype)

        # estimate the information gain
        est_ig = self.updater_poisson.expected_information_gain(expparams)

        # see if they roughly match
        assert_almost_equal(est_ig, TestInformationGain.POISSON_IG, decimal=1)


    def test_continuous_outcomes_ig(self):
        # The gaussian model has a (uncountably) infinite number of outcomes. Test the
        # ig calculation in this case. 

        expparams = self.TAU_EXPPARAMS.astype(self.gaussian_model.expparams_dtype)

        # estimate the ig
        est_ig = self.updater_gaussian.expected_information_gain(expparams,use_cached_samples=True)

        #compute the exact ig
        mu, var, var_lik = TestInformationGain.MU, TestInformationGain.VAR, \
                            TestInformationGain.VAR_LIKELIHOOD

        exact_ig = 1./2*np.log(1+(var/var_lik)*expparams['tau']**2)

        assert_almost_equal(est_ig, exact_ig, decimal=1)


class TestFisherInformation(DerandomizedTestCase):
    # Test the implementation of numerical Fisher Information by comparing to 
    # numbers which were derived by doing analytic/numeric
    # integrals of simple models (binomialm, poisson, and gaussian) in 
    # Mathematica. This test trusts that these calculations
    # were done correctly.

    ALPHA = 1
    BETA = 3
    MU = 1.0
    VAR = 10.0
    VAR_LIKELIHOOD = 1.0
    PRIOR_BETA = BetaDistribution(alpha=ALPHA, beta=BETA)
    PRIOR_GAMMA = GammaDistribution(alpha=ALPHA, beta=BETA)
    PRIOR_NORMAL = NormalDistribution(mean=MU,var=VAR)
    N_PARTICLES = 10000
    N_OUTCOME_SAMPLES = 10000
    # Calculated in Mathematica, IG for the binomial model and the given expparams
    FI_MODELPARAMS = np.arange(1,5,dtype=np.float32).reshape(-1,1)
    BIN_FI_MODELPARAMS = np.linspace(0.01,0.99,5)
    NMEAS_EXPPARAMS = np.arange(1, 11, dtype=int)
    
    # Calculated in Mathematica, IG for the poisson model and the given expparams
    TAU_EXPPARAMS = np.arange(1, 11, dtype=int)


    def setUp(self):

        super(TestFisherInformation,self).setUp()
        
        # Set up relevant models.
        self.poisson_model = BasicPoissonModel(num_outcome_samples=TestFisherInformation.N_OUTCOME_SAMPLES)
        self.coin_model = CoinModel()
        self.binomial_model = BinomialModel(self.coin_model)
        self.gaussian_model = BasicGaussianModel(var=TestFisherInformation.VAR_LIKELIHOOD,
            num_outcome_samples=TestFisherInformation.N_OUTCOME_SAMPLES)

        # Set up updaters for these models using particle approximations 
        # of conjugate priors
        self.updater_poisson = SMCUpdater(self.poisson_model,
                TestFisherInformation.N_PARTICLES,TestFisherInformation.PRIOR_GAMMA)
        self.updater_binomial = SMCUpdater(self.binomial_model,
                TestFisherInformation.N_PARTICLES,TestFisherInformation.PRIOR_BETA)
        self.updater_gaussian = SMCUpdater(self.gaussian_model,
                TestFisherInformation.N_PARTICLES,TestFisherInformation.PRIOR_NORMAL)


    def test_finite_outcomes_fi(self):
        # The binomial model has a finite number of outcomes. Test the 
        # ig calculation in this case.

        expparams = self.NMEAS_EXPPARAMS.astype(self.binomial_model.expparams_dtype)
        p = TestFisherInformation.BIN_FI_MODELPARAMS
        # estimate the information gain
        est_fi = self.binomial_model.fisher_information(TestFisherInformation.BIN_FI_MODELPARAMS,expparams)[0,0]
        p = p[:,np.newaxis]
        n = expparams.astype(np.float32)[np.newaxis,:]

        exact_fi = n/((1-p)*p) 
        # see if they roughly match
        
        assert_almost_equal(est_fi,exact_fi, decimal=1)

    def test_infinite_outcomes_fi(self):
        # The poisson model has a (countably) infinite number of outcomes. Test the 
        # ig calculation in this case.

        expparams = self.TAU_EXPPARAMS.astype(self.poisson_model.expparams_dtype)

        # estimate the information gain
        est_fi = self.poisson_model.fisher_information(TestFisherInformation.FI_MODELPARAMS,expparams)[0,0]

        tau = expparams.astype(np.float32)[np.newaxis,:]
        lam = TestFisherInformation.FI_MODELPARAMS


        exact_fi = tau/lam

        # see if they roughly match
        assert_almost_equal(est_fi, exact_fi, decimal=0)


    def test_continuous_outcomes_fi(self):
        # The gaussian model has a (uncountably) infinite number of outcomes. Test the
        # ig calculation in this case. 

        expparams = self.TAU_EXPPARAMS.astype(self.gaussian_model.expparams_dtype)

        # estimate the ig
        est_fi = self.gaussian_model.fisher_information(TestFisherInformation.FI_MODELPARAMS,expparams)[0,0]
        
        mu, var, var_lik = TestFisherInformation.MU, TestFisherInformation.VAR, \
                            TestFisherInformation.VAR_LIKELIHOOD
        
        exact_fi = expparams.astype(np.float32)**2/var_lik
        

        
        # compare against mean est_fi
        assert_almost_equal(np.mean(est_fi,axis=0), exact_fi, decimal=0)