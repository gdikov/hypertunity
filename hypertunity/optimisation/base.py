# -*- coding: utf-8 -*-
"""Defines the API of every optimiser and implements common logic."""

import abc

from typing import List, Dict, Tuple
from dataclasses import dataclass

from hypertunity.optimisation.domain import Domain, Sample


__all__ = [
    "EvaluationScore",
    "HistoryPoint",
    "Optimiser",
    "Optimizer"
]


@dataclass
class EvaluationScore:
    """A tuple of the evaluation value of the objective and a variance if known."""
    value: float
    variance: float = 0.0


@dataclass
class HistoryPoint:
    """A tuple of a `Sample` at which the objective has been evaluated and the corresponding metrics.
    The latter is a mapping of a metric name to an `EvaluationScore`.
    """
    sample: Sample
    metrics: Dict[str, EvaluationScore]


class Optimiser:
    """Abstract `Optimiser` to be implemented by all subtypes in this package.

    Every `Optimiser` can be run for one single step at a time using the `run_step` method.
    Since the `Optimiser` does not perform the evaluation of the objective function but only
    proposes values from its domain, evaluation history can be supplied via the `update` method.
    The history can be forgotten and the `Optimiser` brought to the initial state via the `reset`
    """
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

    def update(self, x, fx, **kwargs):
        """Update the optimiser's history track.

        Args:
            x: `Sample`, one sample of the domain of the objective function.
            fx: `EvaluationScore`, the evaluation score of the objective at `x`
        """
        if isinstance(x, Sample) and isinstance(fx, EvaluationScore):
            self.history.append(HistoryPoint(sample=x, metrics={"score": fx}))
        elif isinstance(x, (List, Tuple)) and isinstance(fx, (List, Tuple)) and len(x) == len(fx):
            self.history.extend([HistoryPoint(sample=i, metrics=j) for i, j in zip(x, fx)])
        else:
            raise ValueError("Update values for `x` and `f(x)` must be either "
                             "`Sample` and `EvaluationScore` or a list thereof.")

    def reset(self):
        """Reset the optimiser to the initial state."""
        self._history.clear()


Optimizer = Optimiser
