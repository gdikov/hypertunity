import os
import sys
from typing import Any, Dict, List, Union

import tinydb

from hypertunity import utils
from hypertunity.domain import Domain, Sample
from hypertunity.optimisation.base import HistoryPoint

from .base import Reporter

try:
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
except ImportError as err:
    raise ImportError("Install tensorflow>=1.14 and tensorboard>=1.14 "
                      "to support the HParams plugin.") from err


__all__ = [
    "Tensorboard"
]

EAGER_MODE = tf.executing_eagerly()
session_builder = tf.compat.v1.Session
if str(tf.version.VERSION) < "2.":
    summary_file_writer = tf.compat.v2.summary.create_file_writer
    summary_scalar = tf.compat.v2.summary.scalar
else:
    summary_file_writer = tf.summary.create_file_writer
    summary_scalar = tf.summary.scalar


class Tensorboard(Reporter):
    """A :class:`Reporter` subclass to visualise the results in Tensorboard.

    It utilises Tensorboard's HParams plugin as a dashboard for the summary of
    the optimisation. This class prepares and creates entries with the scalar
    data of the experiment trials, containing the domain sample and the
    corresponding metrics.

    Notes:
        The user is responsible for launching TensorBoard in the browser.
    """

    def __init__(self, domain: Domain, metrics: List[str], logdir: str,
                 primary_metric: str = "",
                 database_path: str = None):
        """Initialise the TensorBoard reporter.

        Args:
            domain: :class:`Domain`. The domain to which all evaluated samples belong.
            metrics: :obj:`List[str]`. The names of the metrics.
            logdir: :obj:`str`. Path to a folder for storing the Tensorboard events.
            primary_metric: (optional) :obj:`str`. Primary metric from `metrics`.
                This is used by the :py:meth:`format` method to determine the
                sorting column and the best value. Default is the first one.
            database_path: (optional) :obj:`str`. The path to the database for
                storing experiment history on disk. Default is in-memory storage.
        """
        super(Tensorboard, self).__init__(
            domain, metrics, primary_metric, database_path
        )
        self._hparams_domain = self._convert_to_hparams_domain(self.domain)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self._logdir = logdir
        self._experiment_counter = 0
        self._set_up()
        print(f"Run 'tensorboard --logdir={logdir}' to launch "
              f"the visualisation in TensorBoard", file=sys.stderr)

    @staticmethod
    def _convert_to_hparams_domain(domain: Domain) -> Dict[str, hp.HParam]:
        hparams = {}
        for var_name, dim in domain.flatten().items():
            dim_type = Domain.get_type(dim)
            joined_name = utils.join_strings(var_name, join_char="/")
            if dim_type == Domain.Continuous:
                hp_dim_type = hp.RealInterval
                vals = list(map(float, dim))
            elif dim_type in [Domain.Discrete, Domain.Categorical]:
                hp_dim_type = hp.Discrete
                vals = (dim,)
            else:
                raise TypeError(
                    f"Cannot map subdomain of type {dim_type} "
                    f"to a known HParams domain."
                )
            hparams[joined_name] = hp.HParam(joined_name, hp_dim_type(*vals))
        return hparams

    def _convert_to_hparams_sample(self, sample: Sample) -> Dict[hp.HParam, Any]:
        hparams = {}
        for name, val in sample:
            joined_name = utils.join_strings(name, join_char="/")
            hparams[self._hparams_domain[joined_name]] = val
        return hparams

    def _set_up(self):
        with summary_file_writer(self._logdir).as_default():
            hp.hparams_config(
                hparams=self._hparams_domain.values(),
                metrics=[hp.Metric(m) for m in self.metrics])

    @staticmethod
    def _log_tf_eager_mode(params, metrics, full_experiment_dir):
        """Log in eager mode."""
        with summary_file_writer(full_experiment_dir).as_default():
            hp.hparams(params)
            for metric_name, metric_value in metrics.items():
                summary_scalar(metric_name, metric_value.value, step=1)

    @staticmethod
    def _log_tf_graph_mode(params, metrics, full_experiment_dir):
        """Log in legacy graph execution mode with session creation."""
        with summary_file_writer(full_experiment_dir).as_default() as fw, session_builder() as sess:
            sess.run(fw.init())
            sess.run(hp.hparams(params))
            for metric_name, metric_value in metrics.items():
                sess.run(summary_scalar(metric_name, metric_value.value, step=1))
            sess.run(fw.flush())

    def _log_history_point(self, entry: HistoryPoint, experiment_dir: str = None):
        """Create an entry for a :class:`HistoryPoint` in Tensorboard.

        Args:
            entry: :class:`HistoryPoint`. The sample and evaluation metrics to log.
            experiment_dir: (optional) :obj:`str`. The directory name where to
                store all experiment related data. It will be prefixed by the
                `logdir` path which is provided on initialisation of the
                :class:`Tensorboard` object. Default is 'experiment_[number]'.
        """
        converted = self._convert_to_hparams_sample(entry.sample)
        if not experiment_dir:
            experiment_dir = f"experiment_{str(self._experiment_counter)}"
            self._experiment_counter += 1
        full_experiment_dir = os.path.join(self._logdir, experiment_dir)
        if EAGER_MODE:
            self._log_tf_eager_mode(converted, entry.metrics, full_experiment_dir)
        else:
            self._log_tf_graph_mode(converted, entry.metrics, full_experiment_dir)

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
        super(Tensorboard, self).from_database(database, table)
        for doc in self._db_default_table:
            history_point = self._convert_doc_to_history(doc)
            self._log_history_point(history_point)
