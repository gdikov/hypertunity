import abc
import datetime
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tinydb

from hypertunity.domain import Domain, Sample
from hypertunity.optimisation.base import EvaluationScore, HistoryPoint

__all__ = [
    "Reporter"
]

HistoryEntryType = Union[
    HistoryPoint,
    Tuple[Sample, Union[float, Dict[str, float], Dict[str, EvaluationScore]]]
]


class Reporter:
    """Abstract class :class:`Reporter` for result visualisation."""

    def __init__(self, domain: Domain,
                 metrics: List[str],
                 primary_metric: str = "",
                 database_path: str = None):
        """Initialise the base reporter with domain and metrics.

        Args:
            domain: A :class:`Domain` from which all evaluated samples are drawn.
            metrics: :obj:`List[str]` with names of the metrics used during
                evaluation.
            primary_metric: (optional) :obj:`str` primary metric from `metrics`.
                This is used to determine the best sample. Defaults to the first one.
            database_path: (optional) :obj:`str` path to the database for
                storing experiment history on disk. Defaults to in-memory storage.
        """
        self.domain = domain
        if not metrics:
            self.metrics = ["score"]
        else:
            self.metrics = metrics
        if not primary_metric:
            self.primary_metric = self.metrics[0]
        else:
            self.primary_metric = primary_metric

        self._default_table_name = f"trial_{datetime.datetime.now().isoformat()}"
        if database_path is not None:
            if not os.path.exists(database_path):
                os.makedirs(database_path)
            db_path = os.path.join(database_path, "db.json")
            self._db = tinydb.TinyDB(
                db_path,
                sort_keys=True,
                indent=4,
                separators=(',', ': ')
            )
        else:
            from tinydb.storages import MemoryStorage
            self._db = tinydb.TinyDB(storage=MemoryStorage,
                                     default_table=self._default_table_name)
        self._db_default_table = self._db.table(self._default_table_name)

    @property
    def database(self):
        """Return the logging database."""
        return self._db

    @property
    def default_database_table(self):
        """Return the default database table name."""
        return self._default_table_name

    def log(self, entry: HistoryEntryType, **kwargs: Any):
        """Create an entry for an optimisation history point in the
        :class:`Reporter`.

        Args:
            entry: :class:`HistoryPoint` or :obj:`Tuple[Sample, Dict]`.
                The history point to log. If given as a tuple of :class:`Sample`
                instance and a mapping from metric names to results, the
                variance of the evaluation noise can be supplied by adding
                an entry in the dict with the metric name and the suffix '_var'.
            **kwargs: (optional) :obj:`Any`. Additional arguments for the
                logging implementation in a subclass.

        Keyword Args:
            meta: (optional) additional information to be logged in the database
                for this entry.
        """
        if isinstance(entry, Tuple):
            log_fn = self._log_tuple
        elif isinstance(entry, HistoryPoint):
            self._add_to_db(entry, kwargs.pop("meta", None))
            log_fn = self._log_history_point
        else:
            raise TypeError(
                "The history point can be either a tuple or a "
                "`HistoryPoint` type object."
            )
        log_fn(entry, **kwargs)

    def _log_tuple(self, entry: Tuple, **kwargs):
        """Helper function to convert the history entry from tuple to
        :class:`HistoryPoint` and then log it using the overridden method
        :method:`_log_history_point`.
        """
        if not (len(entry) == 2 and isinstance(entry[0], Sample)
                and isinstance(entry[1], (Dict, EvaluationScore, float))):
            raise ValueError(f"Malformed history entry tuple: {entry}.")
        sample, metrics_obj = entry
        if isinstance(metrics_obj, (float, EvaluationScore)):
            # use default name for score column
            metrics_obj = {self.primary_metric: metrics_obj}
        metrics = {}
        # create a properly formatted metrics dict of type Dict[str, EvaluationScore]
        for name, val in metrics_obj.items():
            if name in metrics:
                continue
            if name.endswith("_var"):
                metric_name = name.rstrip("_var")
                if (metric_name not in metrics_obj
                        or not isinstance(metrics_obj[metric_name], float)):
                    raise ValueError(
                        f"Metrics dict does not contain a proper value "
                        f"for metric {metric_name}."
                    )
                metrics[metric_name] = EvaluationScore(
                    value=metrics_obj[metric_name],
                    variance=val
                )
            elif isinstance(val, EvaluationScore):
                metrics[name] = val
            elif isinstance(val, float):
                metrics[name] = EvaluationScore(
                    value=val,
                    variance=metrics_obj.get(f"{name}_var", 0.0)
                )
        entry = HistoryPoint(sample=sample, metrics=metrics)
        self._add_to_db(entry, kwargs.pop("meta", None))
        self._log_history_point(entry, **kwargs)

    @abc.abstractmethod
    def _log_history_point(self, entry: HistoryPoint, **kwargs: Any):
        """Abstract method to override.

        Log the :class:`HistoryPoint` entry into the reporter.

        Args:
            entry: :class:`HistoryPoint`. The sample and evaluation metrics to log.
        """
        raise NotImplementedError

    def _add_to_db(self, entry: HistoryPoint, meta: Any = None):
        document = self._convert_history_to_doc(entry)
        if meta is not None:
            document["meta"] = meta
        self._db_default_table.insert(document)

    def get_best(self, criterion: Union[str, Callable] = "max") -> Optional[Dict[str, Any]]:
        """Return the entry from the database which corresponds to the best
        scoring experiment.

        Args:
            criterion: :obj:`str` or :obj:`Callable`. The function used to
                determine whether the highest or lowest score is requested. If
                several evaluation metrics are present, then a custom `criterion`
                must be supplied.

        Returns:
            JSON object or `None` if the database is empty. The content of the
            database for the best experiment.
        """
        if not self._db_default_table:
            return None
        if isinstance(criterion, str):
            predefined = {"max": max, "min": min}
            if criterion not in predefined:
                raise ValueError(
                    f"Unknown criterion for finding best experiment. "
                    f"Select one from {list(predefined.keys())} "
                    f"or supply a custom function."
                )
            selection_fn = predefined[criterion]
        elif isinstance(criterion, Callable):
            selection_fn = criterion
        else:
            raise TypeError("The criterion must be of type str or Callable.")
        return self._get_best_from_db(selection_fn)

    def _get_best_from_db(self, selection_fn: Callable):
        best_entry = self._db_default_table.get(doc_id=1)
        best_score = best_entry["metrics"][self.primary_metric]["value"]
        for entry in self._db_default_table:
            current_score = entry["metrics"][self.primary_metric]["value"]
            new_score = selection_fn(current_score, best_score)
            if new_score != best_score:
                best_entry = entry
                best_score = new_score
        return best_entry

    def from_history(self, history: List[HistoryEntryType]):
        """Load the reporter with data from an entry of evaluations.

        Args:
            history: :obj:`List[HistoryPoint]` or :obj:`Tuple`. The sequence of
                evaluations comprised of samples and metrics.
        """
        for h in history:
            self.log(h)

    def from_database(self, database: Union[str, tinydb.TinyDB], table: str = None):
        """Load history from a database supplied as a path to a file or a
        :obj:`tinydb.TinyDB` object.

        Args:
            database: :obj:`str` or :obj:`tinydb.TinyDB`. The database to load.
            table: (optional) :obj:`str`. The table to load from the database.
                This argument is not required if the database has only one table.

        Raises:
            :class:`ValueError`: if the database contains more than one table
                and `table` is not given.
        """
        if isinstance(database, str):
            db = tinydb.TinyDB(database, sort_keys=True, indent=4, separators=(',', ': '))
        elif isinstance(database, tinydb.TinyDB):
            db = database
        else:
            raise TypeError("The database must be of type str or tinydb.TinyDB.")
        if len(db.tables()) > 1 and table is None:
            raise ValueError(
                "Ambiguous database with multiple tables. "
                "Specify a table name."
            )
        if table is None:
            table = list(db.tables())[0]
        self._db = db
        self._db_default_table = self._db.table(table)

    def to_history(self, table: str = None) -> List[HistoryPoint]:
        """Export the reporter logged history from a database table to an
        optimiser-friendly history.

        Args:
            table: (optional) :obj:`str`. The name of the table to export.
                Defaults to the one created during reporter initialisation.

        Returns:
            A list of :class:`HistoryPoint` objects which can be loaded into
            an :class:`Optimiser` instance.
        """
        history = []
        if table is None:
            default_table = self._db_default_table
        else:
            default_table = self._db.table(table)
        for doc in default_table:
            history.append(self._convert_doc_to_history(doc))
        return history

    @staticmethod
    def _convert_history_to_doc(entry: HistoryPoint) -> Dict:
        db_entry = {
            "sample": entry.sample.as_dict(),
            "metrics": {k: {
                "value": v.value,
                "variance": v.variance
            } for k, v in entry.metrics.items()}
        }
        return db_entry

    @staticmethod
    def _convert_doc_to_history(document: Dict) -> HistoryPoint:
        hist_point = HistoryPoint(
            sample=Sample(document["sample"]),
            metrics={k: EvaluationScore(v["value"], v["variance"])
                     for k, v in document["metrics"].items()}
        )
        return hist_point
