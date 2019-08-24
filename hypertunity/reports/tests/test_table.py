from ..table import TableReporter
from ._common import generate_history


def test_from_history():
    n_samples = 10
    history, domain = generate_history(n_samples)
    rep = TableReporter(domain, metrics=["metric_1", "metric_2"], primary_metric_index=0)
    rep.from_history(history)
    data_history = [[i+1, *list(h.sample.flatten().values()), *list(h.metrics.values())]
                    for i, h in enumerate(history)]
    assert rep.data.tolist() == data_history
