# -*- coding: utf-8 -*-
"""Defines the API of every optimiser and implements common logic.
"""

from ..utils import support_american_style


class BaseOptimiser:
    """Abstract `Optimiser` to be implemented by all subtypes in this package.

    Every `Optimiser` can be run for one single step at a time using the `run_step` method.
    Since the `Optimiser` does not perform the evaluation of the objective function but only
    proposes values from its domain, evaluation history can be supplied via the `update` method.
    The history can be forgotten and the `Optimiser` brought to the initial state via the `reset`
    """
    @support_american_style
    def __init__(self, domain, minimise=True):
        self.domain = domain
        self.minimise = minimise

    def run_step(self, *args, **kwargs):
        raise NotADirectoryError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
