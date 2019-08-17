# -*- coding: utf-8 -*-
import pytest

import hypertunity as ht

from . import script
from ..jobs import Job
from ..local import LocalScheduler


def square(sample: ht.Sample) -> ht.EvaluationScore:
    return ht.EvaluationScore(sample["x"] ** 2)


def _run_jobs(jobs):
    with LocalScheduler(n_parallel=2) as scheduler:
        scheduler.dispatch(jobs)
        results = scheduler.collect(n_results=len(jobs), timeout=5.0)
    assert len(results) == len(jobs)
    assert all([r.id == j.id for r, j in zip(results, jobs)])
    return results


@pytest.mark.timeout(10.0)
def test_local_from_script():
    domain = ht.Domain({"x": (0, 1, 2, 3), "y": [-1., 1.], "z": ("123", "abc")}, seed=7)
    jobs = [Job(task="hypertunity/scheduling/tests/script.py",
                args=(*domain.sample().as_namedtuple(),)) for _ in range(10)]
    results = _run_jobs(jobs)
    assert all([r.data == script.main(*j.args) for r, j in zip(results, jobs)])


@pytest.mark.timeout(10.0)
def test_local_from_fn():
    domain = ht.Domain({"x": [0., 1.]}, seed=7)
    jobs = [Job(task=square, args=(domain.sample(),)) for _ in range(10)]
    results = _run_jobs(jobs)
    assert all([r.data.value == square(*j.args).value for r, j in zip(results, jobs)])
