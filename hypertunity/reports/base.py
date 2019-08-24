import abc

from typing import List, Any

from hypertunity.optimisation.base import HistoryPoint
from hypertunity.optimisation.domain import Domain


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

    @abc.abstractmethod
    def log(self, history: HistoryPoint, **kwargs: Any):
        """Create an entry for a `HistoryPoint` in the reporter.

        Args:
            history: `HistoryPoint`, the sample and evaluation metrics to log.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_history(self, history: List[HistoryPoint]):
        """Load the reporter with data from a history of evaluations.

        Args:
            history: list of `HistoryPoint`, the evaluations comprised of samples and metrics.
        """
        raise NotImplementedError
