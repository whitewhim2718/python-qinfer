#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# models.py: Likelihood models for quantum state and process tomography.
##
# © 2015 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@cgranade.com).
# Based on work with Joshua Combes (joshua.combes@gmail.com).
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

# TODO: docstrings!
# TODO: unit tests!

## DOCSTRING #################################################################

"""
"""

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

## IMPORTS ###################################################################

from builtins import range, map

from qinfer import Model

import numpy as np

## EXPORTS ###################################################################
# TODO

## DESIGN NOTES ##############################################################

"""
Bases are always assumed to have exactly one traceful element— in particular,
the zeroth basis element.
"""

## FUNCTIONS #################################################################

# TODO: document, contribute to QuTiP?
def heisenberg_weyl_operators(d=2):
    w = np.exp(2 * np.pi * 1j / d)
    X = qt.Qobj([
        qt.basis(d, (idx + 1) % d).data.todense().view(np.ndarray)[:, 0] for idx in range(d)
    ])
    Z = qt.Qobj(np.diag(w ** np.arange(d)))
    
    return [X**i * Z**j for i in range(d) for j in range(d)]

## CLASSES ###################################################################

# TODO: work from qt.ket2dm(np.random.choice(np.random.choice(heisenberg_weyl_operators(3)).eigenstates()[1])).

# class RandomMeasurementHeuristic(Heuristic):
#     """
#     Samples state tomography experiments which measure against a randomly
#     chosen measurement effect, drawn from a provided list.
#     """
#     def __init__(self, updater, measurment_effects, other_fields=None):
#         self._up = updater
#         self._effects = measurment_effects
#         self._other_fields = {} if other_fields is None else other_fields
#         self._dim = updater.model.base_model.dim
#         self._basis = self.model.base_model.basis

#     def __call__(self):
#         expparams = np.zeros((1,), dtype=self._up.model.expparams_dtype)
#         expparams['meas'][0, :] = self._basis.state_to_modelparams(
#             np.random.choice(self._effects)
#         )

#         for field, value in self._other_fields.iteritems():
#                 expparams[field] = value
        
#         return expparams

# class RandomBasisHeuristic(Heuristic):
#     # TODO: move this one!
#     def __init__(self, updater, other_fields=None):
#         self._up = updater
#         self._other_fields = {} if other_fields is None else other_fields
#         self._dim = updater.model.base_model.dim

#     def __call__(self):
#         expparams = np.zeros((1,), dtype=self._up.model.expparams_dtype)
#         expparams['meas'][0, [0, 1 + np.random.randint(self._dim ** 2 - 1)]] = np.sqrt(self._dim)

#         for field, value in self._other_fields.iteritems():
#                 expparams[field] = value
        
#         return expparams

class TomographyModel(Model):
    def __init__(self, basis, allow_subnormalized=False):
        self._dim = basis.dim
        self._basis = basis
        self._allow_subnormalied = allow_subnormalized
        super(TomographyModel, self).__init__()

    @property
    def dim(self):
        return self._dim
    @property
    def basis(self):
        return self._basis
        

    @property
    def n_modelparams(self):
        return self._dim ** 2

    @property
    def modelparam_names(self):
        return list(map(
            r'\langle\!\langle{} | \rho\rangle\!\rangle'.format,
            self.basis.labels
        ))

    @property
    def is_n_outcomes_constant(self):
        return True

    @property
    def expparams_dtype(self):
        return [
            ('meas', float, self._dim ** 2)
        ]

    def n_outcomes(self, expparams):
        return 2

    def are_models_valid(self, modelparams):
        # This is wrong, but is wrong for the sake of speed.
        # As a future improvement, validity checking needs to
        # be enabled as a non-default option.
        return np.ones((modelparams.shape[0],), dtype=bool)

    def canonicalize(self, modelparams):
        modelparams = np.apply_along_axis(self.trunc_neg_eigs, 1, modelparams)
        # Renormalizes particles if allow_subnormalized=False.
        if not self._allow_subnormalied:
            modelparams = self.renormalize(modelparams)

        return modelparams

    def trunc_neg_eigs(self, particle):
        arr = np.tensordot(particle, self._basis.data.conj(), 1)
        w, v = np.linalg.eig(arr)
        if np.all(w >= 0):
            return particle
        else:
            w[w < 0] = 0
            new_arr = np.dot(v * w, v.conj().T)
            new_particle = np.real(np.dot(self._basis.flat(), new_arr.flatten()))
            assert new_particle[0] > 0
            return new_particle

    def renormalize(self, modelparams):
        # The 0th basis element (identity) should have
        # a value 1 / sqrt{dim}, since the trace of that basis
        # element is fixed to be sqrt{dim} by convention.
        norm = modelparams[:, 0] * np.sqrt(self._dim)
        assert not np.sum(norm == 0)
        return modelparams / norm[:, None]

    def likelihood(self, outcomes, modelparams, expparams):
        super(TomographyModel, self).likelihood(outcomes, modelparams, expparams)

        pr1 = np.empty((modelparams.shape[0], expparams.shape[0]))

        pr1[:, :] = np.einsum(
            'ei,mi->me',
            # This should be the Hermitian conjugate, but since
            # expparams['meas'] is real (that is, since the measurement)
            # is Hermitian, then that's not needed here.
            expparams['meas'],
            modelparams
        )
        np.clip(pr1, 0, 1, out=pr1)

        return Model.pr0_to_likelihood_array(outcomes, 1 - pr1)

class DiffusiveTomographyModel(TomographyModel):
    @property
    def n_modelparams(self):
        return super(DiffusiveTomographyModel, self).n_modelparams + 1

    @property
    def expparams_dtype(self):
        return super(DiffusiveTomographyModel, self).expparams_dtype + [
            ('t', float)
        ]
    

    @property
    def modelparam_names(self):
        return super(DiffusiveTomographyModel, self).modelparam_names + [r'\epsilon']

    def are_models_valid(self, modelparams):
        return np.logical_and(
            super(DiffusiveTomographyModel, self).are_models_valid(modelparams),
            modelparams[:, -1] > 0
        )

    def canonicalize(self, modelparams):
        return np.concatenate([
            super(DiffusiveTomographyModel, self).canonicalize(modelparams[:, :-1]),
            modelparams[:, -1, None]
        ], axis=1)
    def likelihood(self, outcomes, modelparams, expparams):
        return super(DiffusiveTomographyModel, self).likelihood(outcomes, modelparams[:, :-1], expparams)

    def update_timestep(self, modelparams, expparams):
        # modelparams: [n_m, d² + 1]
        # expparams:   [n_e,]

        # eps:         [n_m, 1] * [n_e] → [n_m, n_e, 1]
        eps = (modelparams[:, -1, None] * np.sqrt(expparams['t']))[:, :, None]
        # steps:       [n_m, n_e, 1] * [n_m, 1, d²]
        steps = eps * np.random.randn(*modelparams[:, None, :].shape)
        steps[:, :, [0, -1]] = 0

        raw_modelparams = modelparams[:, None, :] + steps
        # raw_modelparams[:, :, :-1] = np.apply_along_axis(self.trunc_neg_eigs, 2, raw_modelparams[:, :, :-1])
        for idx_experiment in range(len(expparams)):
            raw_modelparams[:, idx_experiment, :] = self.canonicalize(raw_modelparams[:, idx_experiment, :])
        return raw_modelparams.transpose((0, 2, 1))

