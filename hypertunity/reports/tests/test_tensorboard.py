import os
import tempfile

from ._common import generate_history
from ..tensorboard import Tensorboard


def test_from_to_history():
    n_samples = 10
    history, domain = generate_history(n_samples)
    with tempfile.TemporaryDirectory() as tmp_dir:
        rep = Tensorboard(domain, metrics=["metric_1", "metric_2"], logdir=tmp_dir)
        rep.from_history(history)
        assert len(os.listdir(tmp_dir)) == n_samples
        for root, dirs, files in os.walk(tmp_dir):
            assert all(map(lambda x: x.startswith("events.out.tfevents"), files))
        assert rep.to_history() == history


def test_from_tuple_and_history_point():
    hist_point, domain = generate_history(n_samples=1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        rep = Tensorboard(domain, metrics=["metric_1", "metric_2"], logdir=tmp_dir)
        rep.log(hist_point)
        rep.log((domain.sample(), {"metric_1": 1.0, "metric_2": 2.0, "metric_2_var": 3.0}))
        assert len(os.listdir(tmp_dir)) == 2
        for root, dirs, files in os.walk(tmp_dir):
            assert all(map(lambda x: x.startswith("events.out.tfevents"), files))
