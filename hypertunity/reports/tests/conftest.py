import pytest

from hypertunity.domain import Domain
from hypertunity.optimisation.base import EvaluationScore, HistoryPoint


@pytest.fixture(scope="session")
def generated_history():
    domain = Domain({
        "x": [-5., 6.],
        "y": {"sin", "sqr"},
        "z": set(range(4))
    }, seed=7)
    n_samples = 10
    history = [HistoryPoint(sample=domain.sample(),
                            metrics={"metric_1": EvaluationScore(float(i)),
                                     "metric_2": EvaluationScore(i * 2.)})
               for i in range(n_samples)]
    if len(history) == 1:
        history = history[0]
    return history, domain
