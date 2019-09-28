from typing import List, Any

import beautifultable as bt
import numpy as np

from hypertunity import utils
from hypertunity.domain import Domain
from hypertunity.optimisation.base import HistoryPoint

from .base import Reporter


class TableReporter(Reporter):
    """A `Reporter` to print and store a formatted table of the results."""

    def __init__(self, domain: Domain, metrics: List[str],
                 primary_metric: str = "",
                 database_path: str = None):
        """Initialise the table reporter.

        Args:
            domain: `Domain`, the domain to which all evaluated samples belong.
            metrics: list of str, the names of the metrics.
            primary_metric: str, optional primary metric from `metrics`.
                This is used by the `format` method to determine the sorting column and the best value.
                Default is the first one.
            database_path: str, the path to the database for storing experiment history on disk.
        """
        super(TableReporter, self).__init__(domain, metrics, primary_metric, database_path)
        self._table = bt.BeautifulTable()
        self._table.set_style(bt.STYLE_SEPARATED)
        dim_names = [".".join(dns) for dns in self.domain.flatten()]
        self._table.column_headers = ["No.", *dim_names, *self.metrics]

    def __str__(self):
        return str(self._table)

    @property
    def data(self) -> np.array:
        """Return the table as a numpy array."""
        return np.array(self._table)

    def _log_history_point(self, entry: HistoryPoint, **kwargs: Any):
        """Create an entry for a `HistoryPoint` in a table.

        Args:
            entry: `HistoryPoint`, the sample and evaluation metrics to log.
        """
        id_ = len(self._table)
        row = [id_ + 1, *entry.sample.flatten().values(), *entry.metrics.values()]
        self._table.append_row(row)

    @utils.support_american_spelling
    def format(self, order: str = "none", emphasise: bool = False) -> str:
        """Format the table and return it as a string.

        Args:
            order: str, order of sorting by the primary metric. Can be "none", "ascending" or "descending".
            emphasise: bool, whether to emphasise (mark yellow and blink if possible) the best experiment.

        Returns:
            The formatted table as a string.
        """
        table_copy = self._table.copy()
        if order not in ["none", "descending", "ascending"]:
            raise ValueError("`order` argument can only be 'ascending' or 'descending'.")
        if order != "none":
            table_copy.sort(key=self.primary_metric, reverse=order == "descending")
        if emphasise:
            best_row_ind = int(np.argmax(list(table_copy.get_column(self.primary_metric))))
            emphasised_best_row = map(lambda x: f"\033[33;5;7m{x}\033[0m", table_copy[best_row_ind])
            table_copy.update_row(best_row_ind, emphasised_best_row)
        return str(table_copy)
