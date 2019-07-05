# -*- coding: utf-8 -*-
"""Defines the API of every optimiser and implements common logic."""

import abc

from typing import List
from dataclasses import dataclass

from hypertunity.optimisation.domain import Sample, Domain
from ..utils import support_american_spelling


@dataclass
class EvaluationScore:
    value: float
    variance: float = 0.0


@dataclass
class HistoryPoint:
    sample: Sample
    score: EvaluationScore


class BaseOptimiser:
    """Abstract `Optimiser` to be implemented by all subtypes in this package.

    Every `Optimiser` can be run for one single step at a time using the `run_step` method.
    Since the `Optimiser` does not perform the evaluation of the objective function but only
    proposes values from its domain, evaluation history can be supplied via the `update` method.
    The history can be forgotten and the `Optimiser` brought to the initial state via the `reset`
    """
    @support_american_spelling
    def __init__(self, domain: Domain):
        """Initialise the base optimiser class with a domain and direction of optimisation.

        Args:
            domain: `Domain`, the objective function's optimisation domain.
        """
        self.domain = domain
        self._history: List[HistoryPoint] = []

    @property
    def history(self):
        """Return the accumulated optimisation history."""
        return self._history

    @abc.abstractmethod
    def run_step(self, *args, **kwargs) -> List[Sample]:
        """Perform one step of optimisation and suggest the next sample to evaluate.

        Args:
            *args: optional arguments for the Optimiser.
            **kwargs: optional keyword arguments for the Optimiser.

        Returns:
            A list of `Sample` type objects corresponding to the `self.domain` domain with
            suggested locations to evaluate. Can be more than one if the optimiser supports batched sampling.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Update the optimiser history with a pair of a sample and evaluation score or other data.

        Args:
            *args: optional, data supplied to the optimiser from outside such as evaluation scores.
            **kwargs: optional, additional options for the update procedure.
        """
        raise NotImplementedError

    def reset(self):
        """Reset the optimiser to the initial state."""
        self._history.clear()


BaseOptimizer = BaseOptimiser
