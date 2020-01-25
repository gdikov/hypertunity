"""Defines the API of every optimiser and implements common logic."""

import abc
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from hypertunity.domain import Domain, Sample

__all__ = [
    "EvaluationScore",
    "HistoryPoint",
    "Optimiser",
    "Optimizer",
    "ExhaustedSearchSpaceError"
]


@dataclass(frozen=True, order=True)
class EvaluationScore:
    """A tuple of the evaluation value of the objective function
    and a variance if known.
    """
    value: float
    variance: float = 0.0

    def __str__(self):
        return f"{self.value:.3f} ± {math.sqrt(self.variance):.1f}"


@dataclass(frozen=True)
class HistoryPoint:
    """A tuple of a :class:`Sample` at which the objective has been evaluated
    and the corresponding metrics. The metrics are supplied as :obj:`dict`
    mapping of a :obj:`str` metric name to an :class:`EvaluationScore`.
    """
    sample: Sample
    metrics: Dict[str, EvaluationScore]


class Optimiser:
    """Abstract class :class:`Optimiser` for all optimisers.

    It must be implemented by all subclasses in this package.

    Every :class:`Optimiser` instance can be run for one single step using the
    :py:meth:`run_step` method. The :class:`Optimiser` does not perform the
    evaluation of the objective function but only proposes values from its
    domain. Therefore an evaluation history must be supplied via the
    :py:meth`update` method. The history can be erased and the
    :class:`Optimiser` brought to the initial state via the :py:meth:`reset`
    method.
    """

    DEFAULT_METRIC_NAME = "score"

    def __init__(self, domain: Domain):
        """Initialise the optimiser with a domain.

        Args:
            domain: :class:`Domain`. The domain of the objective function.
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

        This method can be used to warm-start an optimiser.

        Args:
            history: :obj:`List[HistoryPoint]`. New history which will
                **overwrite** the old one.
        """
        self.reset()
        for hp in history:
            self.update(hp.sample, hp.metrics)

    @abc.abstractmethod
    def run_step(self, batch_size, *args: Any, **kwargs: Any) -> List[Sample]:
        """Perform one step of optimisation and suggest the next sample to
        evaluate.

        Args:
            batch_size: (optional) :obj:`int`. The number of samples to
                suggest at once.
            *args: optional arguments for the Optimiser.
            **kwargs: optional keyword arguments for the Optimiser.

        Returns:
            A :obj:`List[Sample]` with the suggested samples to evaluate.
        """
        raise NotImplementedError

    def update(self, x, fx, **kwargs):
        """Update the optimiser's history with new points.

        Args:
            x: :class:`Sample` or :obj:`List[Sample]`. The samples at which the
                objective function has been evaluated.
            fx: :class:`EvaluationScore` or :obj:`List[EvaluationScore]`. The
                evaluation scores at the corresponding samples.
        """
        if isinstance(x, Sample):
            self._update_history(x, fx)
        elif (isinstance(x, Sequence)
              and isinstance(fx, Sequence)
              and len(x) == len(fx)):
            for i, j in zip(x, fx):
                self._update_history(i, j)
        else:
            raise ValueError("Update values for `x` and `f(x)` must be either "
                             "a `Sample` and an evaluation or a list thereof.")

    def _update_history(self, x, fx):
        if isinstance(fx, (float, int)):
            history_point = HistoryPoint(
                sample=x,
                metrics={self.DEFAULT_METRIC_NAME: EvaluationScore(fx)}
            )
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
            raise TypeError(
                "Cannot update history for one sample and multiple evaluations."
                " Use batched update instead and provide a list of samples and "
                "a list of evaluation metrics.")
        self.history.append(history_point)

    def reset(self):
        """Reset the optimiser to the initial state."""
        self._history.clear()


class ExhaustedSearchSpaceError(Exception):
    pass


Optimizer = Optimiser
