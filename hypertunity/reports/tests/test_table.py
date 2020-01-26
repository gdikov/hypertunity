import os
import tempfile

from hypertunity.optimisation.base import EvaluationScore

from ..table import Table


def test_from_to_history(generated_history):
    history, domain = generated_history
    rep = Table(
        domain,
        metrics=["metric_1", "metric_2"],
        primary_metric="metric_1"
    )
    rep.from_history(history)
    data_history = [
        [i + 1, *list(h.sample.flatten().values()), *list(h.metrics.values())]
        for i, h in enumerate(history)
    ]
    assert rep.data.tolist() == data_history
    assert rep.to_history() == history


def test_from_tuple_and_history_point(generated_history):
    history, domain = generated_history
    hist_point = history[0]
    rep = Table(
        domain,
        metrics=["metric_1", "metric_2"],
        primary_metric="metric_1"
    )
    rep.log(hist_point)
    sample = domain.sample()
    rep.log((sample, {"metric_1": 1.0, "metric_2": 2.0, "metric_2_var": 3.0}))
    assert rep.data.tolist() == [
        [1, *list(hist_point.sample.flatten().values()),
         *list(hist_point.metrics.values())],
        [2, *list(sample.flatten().values()),
         EvaluationScore(1.0), EvaluationScore(2.0, 3.0)]
    ]


def test_database_and_get_best(generated_history):
    history, domain = generated_history
    with tempfile.TemporaryDirectory() as db_dir:
        rep = Table(
            domain,
            metrics=["metric_1", "metric_2"],
            database_path=db_dir
        )
        best_meta, best_metrics, best_sample = {}, {}, {}
        best_score = float("-inf")
        for i, hp in enumerate(history):
            rep.log(hp, meta={"id": i})
            if hp.metrics["metric_1"].value > best_score:
                best_meta = {"id": i}
                best_metrics = {k: {"value": v.value, "variance": v.variance}
                                for k, v in hp.metrics.items()}
                best_sample = hp.sample.as_dict()
                best_score = hp.metrics["metric_1"].value

        assert len(rep.database.table(rep.default_database_table)) == len(history)
        best_entry = rep.get_best(criterion="max")
        assert best_entry["meta"] == best_meta
        assert best_entry["metrics"] == best_metrics
        assert best_entry["sample"] == best_sample

        rep2 = Table(domain, metrics=["metric_1", "metric_2"])
        rep2.from_database(rep.database, table=rep.default_database_table)
        rep3 = Table(domain, metrics=["metric_1", "metric_2"])
        rep3.from_database(os.path.join(db_dir, "db.json"),
                           table=rep.default_database_table)

        assert str(rep) == str(rep2) == str(rep3)
        assert rep.get_best() == rep2.get_best() == rep3.get_best()
