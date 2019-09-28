"""Defines the API of every optimiser and implements common logic."""

import abc
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence

from hypertunity.domain import Domain, Sample

__all__ = [
    "EvaluationScore",
    "HistoryPoint",
    "Optimiser",
    "Optimizer"
]


@dataclass(frozen=True, order=True)
class EvaluationScore:
    """A tuple of the evaluation value of the objective and a variance if known."""
    value: float
    variance: float = 0.0

    def __str__(self):
        return f"{self.value:.3f} ± {math.sqrt(self.variance):.1f}"


@dataclass(frozen=True)
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

    DEFAULT_METRIC_NAME = "score"

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

    @history.setter
    def history(self, history: List[HistoryPoint]):
        """Set the optimiser history.

        Args:
            history: list of `HistoryPoint`, the new history which will **overwrite** the old one.
        """
        self.reset()
        for hp in history:
            self.update(hp.sample, hp.metrics)

    @abc.abstractmethod
    def run_step(self, batch_size: int = 1, *args: Any, **kwargs: Any) -> List[Sample]:
        """Perform one step of optimisation and suggest the next sample to evaluate.

        Args:
            batch_size: int, the number of samples to suggest at one step of `run_step`.
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
            fx: `EvaluationScore`, the evaluation score of the objective at `x`.
        """
        if isinstance(x, Sample):
            self._update_history(x, fx)
        elif isinstance(x, Sequence) and isinstance(fx, Sequence) and len(x) == len(fx):
            for i, j in zip(x, fx):
                self._update_history(i, j)
        else:
            raise ValueError("Update values for `x` and `f(x)` must be either "
                             "a `Sample` and an evaluation or a list thereof.")

    def _update_history(self, x, fx):
        if isinstance(fx, (float, int)):
            history_point = HistoryPoint(
                sample=x, metrics={self.DEFAULT_METRIC_NAME: EvaluationScore(fx)})
        elif isinstance(fx, EvaluationScore):
            history_point = HistoryPoint(
                sample=x, metrics={self.DEFAULT_METRIC_NAME: fx})
        elif isinstance(fx, Dict):
            metrics = {}
            for key, val in fx.items():
                if isinstance(val, (float, int)):
                    metrics[key] = EvaluationScore(val)
                else:
                    metrics[key] = val
            history_point = HistoryPoint(sample=x, metrics=metrics)
        else:
            raise TypeError("Cannot update history for one sample and multiple evaluations. "
                            "Use batched update instead and provide a list of samples "
                            "and a list of evaluation metrics.")
        self.history.append(history_point)

    def reset(self):
        """Reset the optimiser to the initial state."""
        self._history.clear()


Optimizer = Optimiser
