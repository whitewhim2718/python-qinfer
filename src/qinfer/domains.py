#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# domains.py: module for domains of model outcomes
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

## IMPORTS ###################################################################

from __future__ import division
from __future__ import absolute_import

from builtins import range
from future.utils import with_metaclass

import numpy as np
import scipy.stats as st
import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

from functools import partial

import abc

from qinfer import utils as u

import warnings

## EXPORTS ###################################################################

__all__ = [
    'Domain',
    'RealDomain',
    'IntegerDomain',
    'IntegerTupleDomain'
]

## FUNCTIONS #################################################################


## EXCEPTIONS ################################################################

class CannotBeEnumeratedException(Exception):
    """
    Custom exception which indicates that a generator for all 
    of the elements of a domain has been asked for when it 
    is not possible to do, for example, when the domain 
    has infinitly many values.
    """
    pass

## ABSTRACT CLASSES AND MIXINS ###############################################

class Domain(with_metaclass(abc.ABCMeta, object)):
    """
    Abstract base class for domains of outcomes of models.
    """

    ## ABSTRACT PROPERTIES ##

    @abc.abstractproperty
    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        pass

    @abc.abstractproperty
    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        pass

    @abc.abstractproperty
    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        pass

    @abc.abstractproperty
    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but 
        no other guarantees; useful for testing purposes. 
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        pass

    ## CONCRETE PROPERTIES ##

    @property
    def is_discrete(self):
        """
        Whether or not the domain has a countable number of values.

        :type: `bool`
        """
        return not self.is_continuous

    ## CONCRETE METHODS ##

    def values(self):
        """
        Returns an iterable object that yields elements of the domain.
        For domains where ``is_finite`` is ``True``, all elements 
        of the domain will be yielded exactly once.

        :rtype: something iterable.
        """
        if not self.is_finite:
            raise CannotBeEnumeratedException('All values of this domain cannot be given because it is not finite.')
        else
            raise NotImplementedError()

    ## ABSTRACT METHODS ######################################################

    @abc.abstractmethod
    def in_domain(self, point):
        """
        Returns ``True`` or ``False`` depending on whether the given point is in the domain or not.

        :rtype: `bool`
        """
        pass

## CLASSES ###################################################################

class RealDomain(Domain):
    """
    A domain specifying a contiguous (and possibly open ended) subset 
    of the real numbers.

    :param float min: A number specifying the lowest possible value of the 
        domain. If left as `None`, negative infinity is assumed. 
    :param float max: A number specifying the largest possible value of the 
        domain. If left as `None`, negative infinity is assumed.
    """

    def __init__(self, min=None, max=None):
        self._min = min
        self._max = max

    @property
    def min(self):
        """
        Returns the minimum value of the domain. The outcome 
        None is interpreted as negative infinity.

        :rtype: `float`
        """
        return self._min
    @property
    def max(self):
        """
        Returns the maximum value of the domain. The outcome 
        None is interpreted as positive infinity.

        :rtype: `float`
        """
        return self._max
    

    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return True

    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return False

    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return np.float

    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but 
        no other guarantees; useful for testing purposes. 
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        if self.min is not None:
            return np.array([self.min], dtype=self.dtype)
        if self.max is not None:
            return np.array([self.max], dtype=self.dtype)
        else:
            return np.array([0], dtype=self.dtype)

    def in_domain(self, point):
        """
        Returns ``True`` or ``False`` depending on whether the given point is in the domain or not.

        :rtype: `bool`
        """

        return pself._min <= p and self._max >= p

class IntegerDomain(Domain):
    """
    A domain specifying a contiguous (and possibly open ended) subset 
    of the integers.

    :param int min: A number specifying the lowest possible value of the 
        domain. If `None`, negative infinity is assumed. 
    :param int max: A number specifying the largest possible value of the 
        domain. If left as `None`, negative infinity is assumed.
    """

    def __init__(self, min=0, max=None):
        self._min = min
        self._max = max

    @property
    def min(self):
        """
        Returns the minimum value of the domain. The outcome 
        None is interpreted as negative infinity.

        :rtype: `float`
        """
        return self._min
    @property
    def max(self):
        """
        Returns the maximum value of the domain. The outcome 
        None is interpreted as positive infinity.

        :rtype: `float`
        """
        return self._max
    

    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return False

    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return self.min is not None and self.max is not None

    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return np.int

    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but 
        no other guarantees; useful for testing purposes. 
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        if self.min is not None:
            return np.array([self._min], dtype=self.dtype)
        if self.max is not None:
            return np.array([self._max], dtype=self.dtype)
        else:
            return np.array([0], dtype=self.dtype)

    def in_domain(self, point):
        """
        Returns ``True`` or ``False`` depending on whether the given point is in the domain or not.

        :rtype: `bool`
        """

        return np.mod(p,1) == 0 and self._min <= p and self._max >= p

    def values(self):
        """
        Returns an iterable object that yields elements of the domain.
        For domains where ``is_finite`` is ``True``, all elements 
        of the domain will be yielded exactly once.

        :rtype: something iterable.
        """
        if self.max is None or self.min is None:
            return super(IntegerDomain, self).values()
        else:
            return np.arange(self.min, self.max + 1, dtype = self.dtype)

class IntegerTupleDomain(Domain):
    """
    A domain specifying a hyper-cube of k-tuples of integers

    :param int n_elements: The number of elements of a tuple. 
    :param int min: A number specifying the lowest possible value of each 
        element of a tuple in the domain. If left as `None`, negative infinity is assumed. 
    :param int max: A number specifying the largest possible value of each 
        element of a tuple in the domain. If left as `None`, negative infinity is assumed.
    """

    def __init__(self, min=0, max=None, n_elements=2):
        self._n_elements = n_elements
        self._min = min
        self._max = max

    @property
    def min(self):
        """
        Returns the minimum value of the domain. The outcome 
        None is interpreted as negative infinity.

        :rtype: `float`
        """
        return self._min
    @property
    def max(self):
        """
        Returns the maximum value of the domain. The outcome 
        None is interpreted as positive infinity.

        :rtype: `float`
        """
        return self._max
    @property
    def n_elements(self):
        """
        Returns the number of elements of a tuple in the domain.

        :rtype: `float`
        """
        return self._n_elements
    

    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return False

    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return self.min is not None and self.max is not None

    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return np.dtype([('k', np.int, self.n_elements)])

    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but 
        no other guarantees; useful for testing purposes. 
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        if self.min is not None:
            return self.min * np.ones((1,), dtype=self.dtype)
        if self.max is not None:
            return self.max * np.ones((1,), dtype=self.dtype)
        else:
            return np.zeros((1,), dtype=self.dtype)

    def in_domain(self, point):
        """
        Returns ``True`` or ``False`` depending on whether the given point is in the domain or not.

        :rtype: `bool`
        """

        return np.mod(p,1) == 0 and self._min <= p and self._max >= p

    def values(self):
        """
        Returns an iterable object that yields elements of the domain.
        For domains where ``is_finite`` is ``True``, all elements 
        of the domain will be yielded exactly once.

        :rtype: something iterable.
        """
        if self.max is None or self.min is None:
            super(IntegerDomain, self).values()
        else:
            v = np.indices((self.max-self.min+1)*np.ones(self.n_elements))