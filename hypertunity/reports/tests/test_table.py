import tempfile

from hypertunity.optimisation.base import EvaluationScore
from ._common import generate_history
from ..table import TableReporter


def test_from_history():
    n_samples = 10
    history, domain = generate_history(n_samples)
    rep = TableReporter(domain, metrics=["metric_1", "metric_2"], primary_metric="metric_1")
    rep.from_history(history)
    data_history = [[i + 1, *list(h.sample.flatten().values()), *list(h.metrics.values())]
                    for i, h in enumerate(history)]
    assert rep.data.tolist() == data_history


def test_from_tuple_and_history_point():
    hist_point, domain = generate_history(n_samples=1)
    rep = TableReporter(domain, metrics=["metric_1", "metric_2"], primary_metric="metric_1")
    rep.log(hist_point)
    sample = domain.sample()
    rep.log((sample, {"metric_1": 1.0, "metric_2": 2.0, "metric_2_var": 3.0}))
    assert rep.data.tolist() == [
        [1, *list(hist_point.sample.flatten().values()), *list(hist_point.metrics.values())],
        [2, *list(sample.flatten().values()), EvaluationScore(1.0), EvaluationScore(2.0, 3.0)]
    ]


def test_database_and_get_best():
    hist_points, domain = generate_history(n_samples=10)
    with tempfile.TemporaryDirectory() as db_dir:
        rep = TableReporter(domain, metrics=["metric_1", "metric_2"], database_path=db_dir)
        best_meta, best_metrics, best_sample = {}, {}, {}
        best_score = float("-inf")
        for i, hp in enumerate(hist_points):
            rep.log(hp, meta={"id": i})
            if hp.metrics["metric_1"].value > best_score:
                best_meta = {"id": i}
                best_metrics = {k: {"value": v.value, "variance": v.variance}
                                for k, v in hp.metrics.items()}
                best_sample = hp.sample.as_dict()
                best_score = hp.metrics["metric_1"].value

        assert len(rep.database) == len(hist_points)
        best_entry = rep.get_best(criteria="max")
        assert best_entry["meta"] == best_meta
        assert best_entry["metrics"] == best_metrics
        assert best_entry["sample"] == best_sample
