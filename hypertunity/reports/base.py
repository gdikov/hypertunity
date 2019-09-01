import abc
from typing import List, Any, Union, Tuple, Dict

from hypertunity.optimisation.base import HistoryPoint, EvaluationScore
from hypertunity.optimisation.domain import Domain, Sample

HistoryEntryType = Union[
    HistoryPoint,
    Tuple[Sample, Union[float, Dict[str, float], Dict[str, EvaluationScore]]]
]


class Reporter:
    """Abstract `Reporter` class for result visualisation."""

    def __init__(self, domain: Domain, metrics: List[str]):
        """Initialise the base reporter.

        Args:
            domain: `Domain`, the domain to which all evaluated samples belong.
            metrics: list of str, the names of the metrics.
        """
        self.domain = domain
        self.metrics = metrics

    def log(self, entry: HistoryEntryType, **kwargs: Any):
        """Create an entry for an optimisation history point in the reporter.

        Args:
            entry: Can either be of type `HistoryPoint` or a tuple of `Sample` and [metric name] -> [results] dict.
                In the latter case, a variance of the evaluation noise can be supplied by adding an entry in the dict
                with the [metric name] and a suffix '_var'.
        """
        if isinstance(entry, Tuple):
            log_fn = self._log_tuple
        elif isinstance(entry, HistoryPoint):
            log_fn = self._log_history_point
        else:
            raise TypeError("The history point can be either a tuple or a `HistoryPoint` type object.")
        log_fn(entry, **kwargs)

    def _log_tuple(self, entry: Tuple, **kwargs):
        """Helper function to convert the history entry from tuple to `HistoryPoint` and
        then log it using the overridden method `_log_history_point`.
        """
        if not (len(entry) == 2 and isinstance(entry[0], Sample)
                and isinstance(entry[1], (Dict, EvaluationScore, float))):
            raise ValueError(f"Malformed history entry tuple: {entry}.")
        sample, metrics_obj = entry
        if isinstance(metrics_obj, (float, EvaluationScore)):
            # use default name for score column
            metrics_obj = {"score": metrics_obj}
        metrics = {}
        # create a properly formatted metrics dict of type Dict[str, EvaluationScore]
        for name, val in metrics_obj.items():
            if name in metrics:
                continue
            if name.endswith("_var"):
                metric_name = name.rstrip("_var")
                if metric_name not in metrics_obj or not isinstance(metrics_obj[metric_name], float):
                    raise ValueError(f"Metrics dict does not contain a proper value for metric {metric_name}.")
                metrics[metric_name] = EvaluationScore(value=metrics_obj[metric_name], variance=val)
            elif isinstance(val, EvaluationScore):
                metrics[name] = val
            elif isinstance(val, float):
                metrics[name] = EvaluationScore(value=val, variance=metrics_obj.get(f"{name}_var", 0.0))
        self._log_history_point(HistoryPoint(sample=sample, metrics=metrics), **kwargs)

    @abc.abstractmethod
    def _log_history_point(self, entry: HistoryPoint, **kwargs: Any):
        """Abstract method to override. Log the `HistoryPoint` type entry into the reporter.

        Args:
            entry: `HistoryPoint`, the sample and evaluation metrics to log.
        """
        raise NotImplementedError

    def from_history(self, history: List[HistoryEntryType]):
        """Load the reporter with data from a entry of evaluations.

        Args:
            history: list of `HistoryPoint` or tuples, the sequence of evaluations comprised of samples and metrics.
        """
        for h in history:
            self.log(h)
