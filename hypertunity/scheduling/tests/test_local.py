# -*- coding: utf-8 -*-
import pytest

import hypertunity as ht
from .. import jobs as ht_jobs
from ..jobs import Job
from ..local import LocalScheduler


def square(sample: ht.Sample) -> ht.EvaluationScore:
    return ht.EvaluationScore(sample["x"] ** 2)


def _build_args(sample: ht.Sample):
    return tuple([f"--{name[0]} {val}" for name, val in sample])


def _run_jobs(jobs):
    with LocalScheduler(n_parallel=2) as scheduler:
        scheduler.dispatch(jobs)
        results = scheduler.collect(n_results=len(jobs), timeout=2.0)
    return results


@pytest.mark.timeout(10.0)
def test_local_from_fn():
    domain = ht.Domain({"x": [0., 1.]}, seed=7)
    jobs = [Job(id=i, func=square, args=(domain.sample(),)) for i in range(10)]
    results = _run_jobs(jobs)
    assert len(results) == len(jobs)
    assert all([r.id == j.id for r, j in zip(results, jobs)])
    assert all([r.data.value == square(*j.args).value for r, j in zip(results, jobs)])
    ht_jobs.reset_job_registry()
