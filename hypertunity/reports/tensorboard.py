# -*- coding: utf-8 -*-
import os
import sys
from typing import List, Dict, Any

try:
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
except ImportError as err:
    raise ImportError("Install TensorFlow>=1.14 and tensorboard>=1.14 "
                      "to support the HParams plugin.") from err

from hypertunity.optimisation.domain import Domain, Sample
from hypertunity.optimisation.base import HistoryPoint
from hypertunity import utils

from .base import Reporter

__all__ = [
    "TensorboardReporter"
]

EAGER_MODE = tf.executing_eagerly()
session_builder = tf.compat.v1.Session
if tf.version.VERSION < "2.":
    summary_file_writer = tf.compat.v2.summary.create_file_writer
    summary_scalar = tf.compat.v2.summary.scalar
else:
    summary_file_writer = tf.summary.create_file_writer
    summary_scalar = tf.summary.scalar


class TensorboardReporter(Reporter):
    """Utilise Tensorboard's HParams plugin as a visualisation tool for the summary of the optimisation.
    Prepare and create entries with the scalar data of the experiment trials, containing the domain sample
    and the corresponding metrics.

    The user is responsible for launching TensorBoard.
    """

    def __init__(self, domain: Domain, metrics: List[str], logdir: str):
        """Initialise the TensorBoard reporter.

        Args:
            domain: `Domain`, the domain to which all evaluated samples belong.
            metrics: list of str, the names of the metrics.
            logdir: str, path to a folder for storing the Tensorboard events.
        """
        super(TensorboardReporter, self).__init__(domain, metrics)
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
                vals = dim
            elif dim_type in [Domain.Discrete, Domain.Categorical]:
                hp_dim_type = hp.Discrete
                vals = (dim,)
            else:
                raise TypeError(f"Cannot map subdomain of type {dim_type} to a known HParams domain.")
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

    def log(self, history: HistoryPoint, experiment_dir: str = None):
        """Create an entry for a `HistoryPoint` in Tensorboard.

        Args:
            history: `HistoryPoint`, the sample and evaluation metrics to log.
            experiment_dir: str, the directory name where to store all experiment related data.
                It will be prefixed by the `logdir` path which is provided on initialisation
                of the `TensorboardReporter`.
        """
        converted = self._convert_to_hparams_sample(history.sample)
        if not experiment_dir:
            experiment_dir = f"experiment_{str(self._experiment_counter)}"
            self._experiment_counter += 1
        full_experiment_dir = os.path.join(self._logdir, experiment_dir)
        if EAGER_MODE:
            self._log_tf_eager_mode(converted, history.metrics, full_experiment_dir)
        else:
            self._log_tf_graph_mode(converted, history.metrics, full_experiment_dir)

    def from_history(self, history: List[HistoryPoint]):
        """Create Tensorboard entries for all points in a history of evaluation samples.

        Args:
            history: list of `HistoryPoint`, the optimisation history to visualise.
        """
        for i, h in enumerate(history):
            self.log(h, experiment_dir=str(i))
