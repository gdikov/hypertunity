from typing import Any, List, Union

import beautifultable as bt
import numpy as np
import tinydb

from hypertunity import utils
from hypertunity.domain import Domain
from hypertunity.optimisation.base import HistoryPoint

from .base import Reporter

__all__ = [
    "Table"
]


class Table(Reporter):
    """A :class:`Reporter` subclass to print and store a formatted table of
    the results.
    """

    def __init__(self, domain: Domain,
                 metrics: List[str],
                 primary_metric: str = "",
                 database_path: str = None):
        """Initialise the table reporter with domain and metrics.

        Args:
            domain: A :class:`Domain` from which all evaluated samples are drawn.
            metrics: :obj:`List[str]` with names of the metrics used during evaluation.
            primary_metric: (optional) :obj:`str` primary metric from `metrics`.
                This is used to determine the best sample. Defaults to the first one.
            database_path: (optional) :obj:`str` path to the database for
                storing experiment history on disk. Defaults to in-memory storage.
        """
        super(Table, self).__init__(
            domain, metrics, primary_metric, database_path
        )
        self._table = bt.BeautifulTable()
        self._table.set_style(bt.STYLE_SEPARATED)
        dim_names = [".".join(dns) for dns in self.domain.flatten()]
        self._table.column_headers = ["No.", *dim_names, *self.metrics]

    def __str__(self):
        """Return the string representation of the table."""
        return str(self._table)

    @property
    def data(self) -> np.array:
        """Return the table as a numpy array."""
        return np.array(self._table)

    def _log_history_point(self, entry: HistoryPoint, **kwargs: Any):
        """Create an entry for a :class:`HistoryPoint` in the table.

        Args:
            entry: :class:`HistoryPoint`. The history point to log. If given as
                a tuple of :class:`Sample` instance and a mapping from metric
                names to results, the variance of the evaluation noise can be
                supplied by adding an entry in the dict with the metric name and
                the suffix '_var'.
        """
        id_ = len(self._table)
        row = [id_ + 1,
               *entry.sample.flatten().values(),
               *entry.metrics.values()]
        self._table.append_row(row)

    @utils.support_american_spelling
    def format(self, order: str = "none", emphasise: bool = False) -> str:
        """Format the table and return it as a string.

        Supported formatting is sorting and emphasising of the best result.

        Args:
            order: (optional) :obj:`str`. The order of sorting by the primary
                metric. Can be "none", "ascending" or "descending".
                Defaults to "none".
            emphasise: (optional) :obj:`bool`. Whether to emphasise the best
                experiment by marking it in yellow and blinking if supported.
                Defaults to `False`.

        Returns:
            :obj:`str` of the formatted table.
        """
        table_copy = self._table.copy()
        if order not in ["none", "descending", "ascending"]:
            raise ValueError(
                "`order` argument can only be 'ascending' or 'descending'."
            )
        if order != "none":
            table_copy.sort(
                key=self.primary_metric,
                reverse=order == "descending"
            )
        if emphasise:
            best_row_ind = int(np.argmax(
                list(table_copy.get_column(self.primary_metric))
            ))
            emphasised_best_row = map(
                lambda x: f"\033[33;5;7m{x}\033[0m", table_copy[best_row_ind]
            )
            table_copy.update_row(best_row_ind, emphasised_best_row)
        return str(table_copy)

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
        super(Table, self).from_database(database, table)
        for doc in self._db_default_table:
            history_point = self._convert_doc_to_history(doc)
            self._log_history_point(history_point)
