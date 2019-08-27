import os
import tempfile

from ._common import generate_history
from ..tensorboard import TensorboardReporter


def test_from_history():
    n_samples = 10
    history, domain = generate_history(n_samples)
    with tempfile.TemporaryDirectory() as tmp_dir:
        TensorboardReporter(domain, metrics=["metric_1", "metric_2"], logdir=tmp_dir).from_history(history)
        assert len(os.listdir(tmp_dir)) == n_samples
        for root, dirs, files in os.walk(tmp_dir):
            assert all(map(lambda x: x.startswith("events.out.tfevents"), files))
