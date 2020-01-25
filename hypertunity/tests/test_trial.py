import pytest

from hypertunity import Domain, Trial
from hypertunity.optimisation import RandomSearch
from hypertunity.reports import Table
from hypertunity.scheduling import Job
from hypertunity.scheduling.tests.test_scheduler import run_jobs


def foo(x, y, z):
    return x**2 + y**2 - z**3


@pytest.mark.timeout(60.0)
def test_trial_with_callable():
    domain = Domain({"x": [-1., 1.], "y": [-2, 2], "z": {1, 2, 3, 4}})
    trial = Trial(objective=foo, domain=domain,
                  optimiser="random_search",
                  database_path=None,
                  seed=7, metrics=["score"])
    n_steps = 10
    batch_size = 2
    trial.run(n_steps, batch_size=batch_size, n_parallel=batch_size)

    rs = RandomSearch(domain=domain, seed=7)
    rep = Table(domain, metrics=["score"])
    for i in range(n_steps):
        samples = rs.run_step(batch_size=batch_size, minimise=False)
        results = [foo(*s.as_namedtuple(), ) for s in samples]
        for sample_eval in zip(samples, results):
            rep.log(sample_eval)

    assert len(trial.optimiser.history) == n_steps * batch_size
    assert str(rep.format(order="ascending")) == str(
        trial.reporter.format(order="ascending")
    )


@pytest.mark.timeout(60.0)
def test_trial_with_script():
    domain = Domain({
        "--x": {0, 1, 2, 3},
        "--y": [-1., 1.],
        "--z": {"123", "abc"}
    })
    trial = Trial(objective="hypertunity/scheduling/tests/script.py",
                  domain=domain,
                  optimiser="random_search",
                  database_path=None,
                  seed=7, metrics=["score"])
    batch_size = 4
    trial.run(n_steps=1, batch_size=batch_size, n_parallel=batch_size)

    rs = RandomSearch(domain=domain, seed=7)
    samples = rs.run_step(batch_size=batch_size)
    jobs = [Job(task="hypertunity/scheduling/tests/script.py",
                args=s.as_dict(),
                meta={"binary": "python"}) for s in samples]
    results = [r.data for r in run_jobs(jobs)]
    assert results == [h.metrics["score"].value
                       for h in trial.optimiser.history]
