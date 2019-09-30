import abc
import datetime
import os
from typing import List, Any, Union, Tuple, Dict, Callable, Optional

import tinydb

from hypertunity.domain import Domain, Sample
from hypertunity.optimisation.base import HistoryPoint, EvaluationScore

__all__ = [
    "Reporter"
]

HistoryEntryType = Union[
    HistoryPoint,
    Tuple[Sample, Union[float, Dict[str, float], Dict[str, EvaluationScore]]]
]


class Reporter:
    """Abstract `Reporter` class for result visualisation."""

    def __init__(self, domain: Domain,
                 metrics: List[str],
                 primary_metric: str = "",
                 database_path: str = None):
        """Initialise the base reporter.

        Args:
            domain: `Domain`, the domain to which all evaluated samples belong.
            metrics: list of str, the names of the metrics.
            primary_metric: str, optional primary metric from `metrics`.
                This is used by the `format` method to determine the sorting column and the best value.
                Default is the first one.
            database_path: str, the path to the database for storing experiment history on disk.
        """
        self.domain = domain
        self.metrics = metrics
        self.primary_metric = primary_metric or self.metrics[0]

        table_name = f"trial_{datetime.datetime.now().isoformat()}"
        if database_path is not None:
            if not os.path.exists(database_path):
                os.makedirs(database_path)
            db_path = os.path.join(database_path, "db.json")
            self._db = tinydb.TinyDB(db_path, sort_keys=True, indent=4, separators=(',', ': '))
        else:
            from tinydb.storages import MemoryStorage
            self._db = tinydb.TinyDB(storage=MemoryStorage, default_table=table_name)
        self._db_current_table = self._db.table(table_name)

    @property
    def database(self):
        return self._db_current_table

    def log(self, entry: HistoryEntryType, **kwargs: Any):
        """Create an entry for an optimisation history point in the reporter.

        Args:
            entry: Can either be of type `HistoryPoint` or a tuple of `Sample` and [metric name] -> [results] dict.
                In the latter case, a variance of the evaluation noise can be supplied by adding an entry in the dict
                with the [metric name] and a suffix '_var'.
        Keyword Args:
            meta: Any, optional additional information to be logged in the database for this entry.
        """
        if isinstance(entry, Tuple):
            log_fn = self._log_tuple
        elif isinstance(entry, HistoryPoint):
            self._add_to_db(entry, kwargs.pop("meta", None))
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
        entry = HistoryPoint(sample=sample, metrics=metrics)
        self._add_to_db(entry, kwargs.pop("meta", None))
        self._log_history_point(entry, **kwargs)

    @abc.abstractmethod
    def _log_history_point(self, entry: HistoryPoint, **kwargs: Any):
        """Abstract method to override. Log the `HistoryPoint` type entry into the reporter.

        Args:
            entry: `HistoryPoint`, the sample and evaluation metrics to log.
        """
        raise NotImplementedError

    def _add_to_db(self, entry: HistoryPoint, meta: Any = None):
        document = _convert_history_to_doc(entry)
        if meta is not None:
            document["meta"] = meta
        self._db_current_table.insert(document)

    def get_best(self, criterion: Union[str, Callable] = "max") -> Optional[Dict[str, Any]]:
        """Return the entry from the database which corresponds to the best scoring experiment.

        Args:
            criterion: str or Callable, the function used to determine whether the highest or lowest score is
                requested. If the evaluation metrics are more than one, then a custom `criteria` must be supplied.

        Returns:
            The content of the database for the best experiment, as JSON object or `None` if the database is empty.
        """
        if not self._db_current_table:
            return None
        if isinstance(criterion, str):
            predefined = {"max": max, "min": min}
            if criterion not in predefined:
                raise ValueError(f"Unknown criterion for finding best experiment. "
                                 f"Select one from {list(predefined.keys())} "
                                 f"or supply a custom function.")
            selection_fn = predefined[criterion]
        elif isinstance(criterion, Callable):
            selection_fn = criterion
        else:
            raise TypeError("The criterion must be of type str or Callable.")
        return self._get_best_from_db(selection_fn)

    def _get_best_from_db(self, selection_fn: Callable):
        best_entry = self._db_current_table.get(doc_id=1)
        best_score = best_entry["metrics"][self.primary_metric]["value"]
        for entry in self._db_current_table:
            current_score = entry["metrics"][self.primary_metric]["value"]
            new_score = selection_fn(current_score, best_score)
            if new_score != best_score:
                best_entry = entry
                best_score = new_score
        return best_entry

    def from_history(self, history: List[HistoryEntryType]):
        """Load the reporter with data from an entry of evaluations.

        Args:
            history: list of `HistoryPoint` or tuples, the sequence of evaluations comprised of samples and metrics.
        """
        for h in history:
            self.log(h)


def _convert_history_to_doc(entry: HistoryPoint) -> Dict:
    db_entry = {
        "sample": entry.sample.as_dict(),
        "metrics": {k: {"value": v.value, "variance": v.variance} for k, v in entry.metrics.items()}
    }
    return db_entry
