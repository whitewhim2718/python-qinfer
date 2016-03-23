..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _robustness_guide:
    
.. currentmodule:: qinfer.derived_models

Robustness Testing
==================

Introduction
------------

In developing statistical inference applications, it is essential to test the
robustness of one's software to errors and noise of various kinds. Thus,
QInfer provides tools to do so by corrupting likelihood calculations in various
realistic ways.

Modeling Statistical and Sampling Error
---------------------------------------

A basic kind of robustness testing can be performed by using
:class:`PoisonedModel`, which adds noise to a model's :meth:`~qinfer.abstract_model.Model.likelihood`
method in such a way as to simulate sampling errors incurred in LFPE approaches
[FG13]_. The noise that :class:`PoisonedModel` adds can be specified as the
tolerance of an adaptive likelihood estimation (ALE) step [FG13]_, or as the number
of samples and hedging used for a hedged maximum likelihood estimator of
the likelihood [FB12]_. In either case, the requested noise is added to the
likelihood reported by the underlying model, such that

.. math::

    \widehat{\Pr}(d | \vec{x}; \vec{e}) = \Pr(d | \vec{x}; \vec{e}) + \epsilon,
    
where :math:`\widehat{\Pr}` is the reported estimate of the true likelihood.

For example, to simulate using adaptive likelihood estimation to reach a
threshold tolerance of 0.01:

>>> from qinfer.test_models import SimplePrecessionModel
>>> from qinfer.derived_models import PoisonedModel
>>> model = PoisonedModel(SimplePrecessionModel(), tol=0.01)



Testing Faulty Simulators
-------------------------

TODO

