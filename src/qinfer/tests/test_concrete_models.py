#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_concrete_models.py: Checks that built-in instances of Model work properly.
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

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.
from future.utils import with_metaclass

## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import numpy.lib.recfunctions as rfn

from qinfer.tests.base_test import (
    DerandomizedTestCase, 
    ConcreteDifferentiableModelTest,
    ConcreteModelTest,
    ConcreteSimulatableTest,
    MockDirectView,
    MockModel
)
import abc
from qinfer import (
    SimplePrecessionModel, SimpleInversionModel,
    CoinModel, NoisyCoinModel, NDieModel,
    BasicPoissonModel, ExponentialPoissonModel,
    BasicGaussianModel, ExponentialGaussianModel,
    RandomizedBenchmarkingModel,
    ReferencedPoissonModel,
    PoisonedModel, BinomialModel, MultinomialModel,
    MLEModel, RandomWalkModel,
    ProductDistribution,
    NormalDistribution, UniformDistribution,
    BetaDistribution, GammaDistribution, 
    PostselectedDistribution,
    ConstrainedSumDistribution,
    DirectViewParallelizedModel
)
from qinfer.ale import ALEApproximateModel
from qinfer.tomography import TomographyModel, DiffusiveTomographyModel, pauli_basis, GinibreDistribution
from qinfer.utils import check_qutip_version

import unittest

# We skip this module entirely under Python 3.3, since there are a lot of
# spurious known failures that still need to be debugged.

import sys
if sys.version_info.major == 3 and sys.version_info.minor <= 3:
    raise unittest.SkipTest("Skipping known failures on 3.3.")

## SIMPLE TEST MODELS #########################################################

class TestSimplePrecessionModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests SimplePrecessionModel.
    """

    def instantiate_model(self):
        return SimplePrecessionModel()
    def instantiate_prior(self):
        return UniformDistribution(np.array([[10,12]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestSimpleInversionModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests SimpleInversionModel.
    """

    def instantiate_model(self):
        return SimpleInversionModel()
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        ws = np.linspace(0,0.5,10, dtype=[('w_','float')])
        ts = np.linspace(0,5,10, dtype=[('t','float')])
        return rfn.merge_arrays([ts, ws])

class TestCoinModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests CoinModel.
    """

    def instantiate_model(self):
        return CoinModel()
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        # only the length of this array matters since CoinModel has no expparams.
        return np.ones((10,),dtype=int)

class TestNoisyCoinModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests NoisyCoinModel.
    """

    def instantiate_model(self):
        return NoisyCoinModel()
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        alphas = (0.1 * np.ones((10,))).astype([('alpha','float')])
        betas = np.linspace(0,0.5,10, dtype=[('beta','float')])
        return rfn.merge_arrays([alphas,betas])

class TestNDieModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests NoisyCoinModel.
    """

    def instantiate_model(self):
        return NDieModel(n=6)
    def instantiate_prior(self):
        unif = UniformDistribution(np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]))
        return ConstrainedSumDistribution(unif, desired_total=1)
    def instantiate_expparams(self):
        return np.arange(10).astype(self.model.expparams_dtype)

## SIMPLE TEST MODELS -- INFINITE OUTCOMES ####################################

class TestBasicPoissonModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BasicPoissonModel.
    """

    def instantiate_model(self):
        return BasicPoissonModel()
    def instantiate_prior(self):
        return GammaDistribution(mean=100, var=10)
    def instantiate_expparams(self):
        return np.linspace(1,100,10).astype(self.model.expparams_dtype)

class TestExponentialPoissonModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests ExponentialPoissonModel.
    """

    def instantiate_model(self):
        return ExponentialPoissonModel()
    def instantiate_prior(self):
        return GammaDistribution(mean=100, var=10)
    def instantiate_expparams(self):
        return np.linspace(1,100,10).astype(self.model.expparams_dtype)

class TestBasicGaussianModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests BasicGaussianModel with a fixed known variance.
    """

    def instantiate_model(self):
        return BasicGaussianModel(var=1.2)
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,6]]))
    def instantiate_expparams(self):
        # only the length of this should matter as the model has no expparams
        return np.ones(10).astype(self.model.expparams_dtype)

class TestBasicGaussianModelUnknownVariance(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests BasicGaussianModel where the variance is an extra model param.
    """

    def instantiate_model(self):
        return BasicGaussianModel(var=None)
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,6],[0,1]]))
    def instantiate_expparams(self):
        # only the length of this should matter as the model has no expparams
        return np.ones(10).astype(self.model.expparams_dtype)

class TestExponentialGaussianModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests ExponentialGaussianModel with a fixed known variance.
    """

    def instantiate_model(self):
        return ExponentialGaussianModel(var=1.2)
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,6]]))
    def instantiate_expparams(self):
        return np.linspace(1,100,10).astype(self.model.expparams_dtype)

class TestExponentialGaussianModelUnknownVariance(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests ExponentialGaussianModel where the variance is an extra model param.
    """

    def instantiate_model(self):
        return ExponentialGaussianModel(var=None)
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,6],[0,1]]))
    def instantiate_expparams(self):
        return np.linspace(1,100,10).astype(self.model.expparams_dtype)



        

## TOMOGRAPHY MODELS ##########################################################

@unittest.skipIf(not check_qutip_version('3.2'), 'This test requires qutip 3.2 or higher to run.')
class TestTomographyModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests TomographyModel.
    """

    def instantiate_model(self):
        basis = pauli_basis(nq=2)
        return TomographyModel(basis=basis)
    def instantiate_prior(self):
        basis = pauli_basis(nq=2)
        return GinibreDistribution(basis)
    def instantiate_expparams(self):
        # 10 different random measurements, each measurement 
        # is an operator expressed in the 2-qubit pauli basis.
        eps = np.random.rand(10, 2 ** 4)
        # now we need to convert to fancy data type by putting 
        # the second index into the 'meas' structure
        eps = eps.view(dtype=self.model.expparams_dtype).squeeze(-1)
        return eps

## RB MODELS ##################################################################

class TestRBModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests RandomizedBenchmarkingModel without interleaving.
    """

    def instantiate_model(self):
        return RandomizedBenchmarkingModel(interleaved=False)
    def instantiate_prior(self):
        return PostselectedDistribution(
            UniformDistribution(np.array([[0,1],[0,1],[0,1]])),
            self.model
        )
    def instantiate_expparams(self):
        ms = np.arange(10).astype(self.model.expparams_dtype)
        return ms

class TestIRBModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests RandomizedBenchmarkingModel with interleaving.
    """

    def instantiate_model(self):
        return RandomizedBenchmarkingModel(interleaved=True)
    def instantiate_prior(self):
        return PostselectedDistribution(
            UniformDistribution(np.array([[0,1],[0,1],[0,1],[0,1]])),
            self.model
        )
    def instantiate_expparams(self):
        # sequential sequences
        ms = np.arange(10).astype([('m','uint')])
        isref = np.random.rand(10).round().astype([('reference',bool)])
        return rfn.merge_arrays([ms, isref])

## DERIVED MODELS #############################################################

# not technically a derived model, but should be.
class TestALEApproximateModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests ALEApproximateModel with SimplePrecessionModel as the underlying model
    (underlying model has 1 scalar expparams).
    """

    def instantiate_model(self):
        return ALEApproximateModel(SimplePrecessionModel())
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        ts = np.linspace(0,5,10, dtype=self.model.expparams_dtype)
        return ts

class TestReferencedPoissonModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests ReferencedPoissonModel with CoinModel as the underlying model
    (underlying model has no expparams).
    """

    def instantiate_model(self):
        return ReferencedPoissonModel(CoinModel())
    def instantiate_prior(self):
        p_dist = BetaDistribution(mean=0.5, var=0.1)
        ref_dist = PostselectedDistribution(
            UniformDistribution(np.array([[6000,6100],[900,1000]])),
            self.model
        )
        return ProductDistribution(p_dist, ref_dist)
    def instantiate_expparams(self):
        b = ReferencedPoissonModel.BRIGHT
        d = ReferencedPoissonModel.DARK
        s = ReferencedPoissonModel.SIGNAL
        modes = np.repeat(np.array([b,d,s]),4).astype([('mode','int')])
        # the ps values don't matter since CoinValue has no expparams
        ps = np.arange(12).astype([('p','float')])
        return rfn.merge_arrays([modes, ps])

class TestReferencedPoissonModel2(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests ReferencedPoissonModel with NoisyCoinModel as the underlying model
    (underlying model has 2 expparams).
    """

    def instantiate_model(self):
        return ReferencedPoissonModel(CoinModel())
    def instantiate_prior(self):
        p_dist = BetaDistribution(mean=0.5, var=0.1)
        ref_dist = PostselectedDistribution(
            UniformDistribution(np.array([[6000,6100],[900,1000]])),
            self.model
        )
        return ProductDistribution(p_dist, ref_dist)
    def instantiate_expparams(self):
        b = ReferencedPoissonModel.BRIGHT
        d = ReferencedPoissonModel.DARK
        s = ReferencedPoissonModel.SIGNAL
        modes = np.repeat(np.array([b,d,s]),4).astype([('mode','int')])
        # the ps values don't matter since CoinValue has no expparams
        alphas = (0.1 * np.ones((10,))).astype([('alpha','float')])
        betas = np.linspace(0,0.5,10, dtype=[('beta','float')])
        return rfn.merge_arrays([alphas, betas, modes])

class TestBinomialModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BinomialModel with CoinModel as the underlying model
    (underlying model has no expparams).
    """

    def instantiate_model(self):
        return BinomialModel(CoinModel())
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        return np.arange(100, 120).astype(self.model.expparams_dtype)

class TestBinomialModel1(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BinomialModel with SimplePrecessionModel as the underlying model
    (underlying model has 1 scalar expparams).
    """

    def instantiate_model(self):
        return BinomialModel(SimplePrecessionModel())
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        # the scalar expparam is given name 'x' by BinomialModel
        ts = np.linspace(0,5,10, dtype=[('x','float')])
        nmeas = np.arange(10,20).astype([('n_meas','int')])
        return rfn.merge_arrays([ts,nmeas])

class TestBinomialModel2(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BinomialModel with NoisyCoinModel as the underlying model
    (underlying model has 2 expparams).
    """

    def instantiate_model(self):
        return BinomialModel(NoisyCoinModel())
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        alphas = (0.1 * np.ones((10,))).astype([('alpha','float')])
        betas = np.linspace(0,0.5,10, dtype=[('beta','float')])
        nmeas = np.arange(10,20).astype([('n_meas','int')])
        return rfn.merge_arrays([alphas,betas,nmeas])

class TestMultinomialModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests MultinomialModel with NDieModel as the underlying model
    (underlying model has no expparams).
    """

    def instantiate_model(self):
        return MultinomialModel(NDieModel(n=6))
    def instantiate_prior(self):
        unif = UniformDistribution(np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]))
        return ConstrainedSumDistribution(unif, desired_total=1)
    def instantiate_expparams(self):
        return np.arange(10).astype(self.model.expparams_dtype)

class TestPoisonedModelALE(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests PoisonedModel with SimplePrecessionModel as the underlying model
    in ALE mode.
    """

    def instantiate_model(self):
        return PoisonedModel(
                SimplePrecessionModel(),
                tol = 1e-4
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestPoisonedModelMLE(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests PoisonedModel with SimplePrecessionModel as the underlying model
    in ALE mode.
    """

    def instantiate_model(self):
        return PoisonedModel(
                SimplePrecessionModel(),
                n_samples = 10,
                hedge = 0.01
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestMLEModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests MLEModel with SimplePrecessionModel and 
    a normal distribution at each step.
    """

    def instantiate_model(self):
        return MLEModel(
                SimplePrecessionModel(),
                likelihood_power = 2
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
       return np.arange(10,20).astype(self.model.expparams_dtype)

class TestRandomWalkModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests RandomWalkModel with SimplePrecessionModel and 
    a normal distribution at each step.
    """

    def instantiate_model(self):
        return RandomWalkModel(
                SimplePrecessionModel(),
                step_distribution = NormalDistribution(mean=0.1,var=0.1)
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestDirectViewParallelizedModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests DirectViewParallelizedModel acting on a MockModel, using
    mocked ipyparallel views.
    """

    _old_ipp = None

    def setUp(self):
        super(TestDirectViewParallelizedModel, self).setUp()

        import qinfer.parallel
        self._old_ipp = qinfer.parallel.ipp
        qinfer.parallel.ipp = 'something other than None'
    
    def tearDown(self):
        super(TestDirectViewParallelizedModel, self).tearDown()

        import qinfer.parallel
        qinfer.parallel.ipp = self._old_ipp

    def instantiate_model(self):
        return DirectViewParallelizedModel(
            MockModel(n_mps=2),
            MockDirectView(),
            # Disable the serial threshold to force the parallel
            # model to use our mocked direct view.
            serial_threshold=0
        )
    def instantiate_prior(self):
        return UniformDistribution([[0, 1]] * 2)
    def instantiate_expparams(self):
        return np.array([(10.0, 2)], dtype=MockModel().expparams_dtype)
