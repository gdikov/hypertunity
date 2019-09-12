import os
import tempfile

import pytest

import hypertunity as ht
from . import script
from ..jobs import Job, SlurmJob
from ..scheduler import Scheduler


def square(sample: ht.Sample) -> ht.EvaluationScore:
    return ht.EvaluationScore(sample["x"] ** 2)


def run_jobs(jobs):
    with Scheduler(n_parallel=2) as scheduler:
        scheduler.dispatch(jobs)
        results = scheduler.collect(n_results=len(jobs), timeout=60.0)
    assert len(results) == len(jobs)
    assert all([r.id == j.id for r, j in zip(results, jobs)])
    return results


@pytest.mark.timeout(10.0)
def test_local_from_script_and_function():
    domain = ht.Domain({"x": {0, 1, 2, 3}, "y": [-1., 1.], "z": {"123", "abc"}}, seed=7)
    jobs = [Job(task="hypertunity/scheduling/tests/script.py::main",
                args=(*domain.sample().as_namedtuple(),)) for _ in range(10)]
    results = run_jobs(jobs)
    assert all([r.data == script.main(*j.args) for r, j in zip(results, jobs)])


@pytest.mark.timeout(10.0)
def test_local_from_script_and_cmdline_args():
    domain = ht.Domain({"x": {0, 1, 2, 3}, "y": [-1., 1.], "z": {"123", "abc"}}, seed=7)
    jobs = [Job(task="hypertunity/scheduling/tests/script.py",
                args=(*domain.sample().as_namedtuple(),),
                meta={"binary": "python"}) for _ in range(10)]
    results = run_jobs(jobs)
    assert all([r.data == script.main(*j.args) for r, j in zip(results, jobs)])


@pytest.mark.timeout(10.0)
def test_local_from_script_and_cmdline_named_args():
    domain = ht.Domain({"--x": {0, 1, 2, 3}, "--y": [-1., 1.], "--z": {"acb123", "abc"}}, seed=7)
    jobs = [Job(task="hypertunity/scheduling/tests/script.py",
                args=domain.sample().as_dict(),
                meta={"binary": "python"}) for _ in range(10)]
    results = run_jobs(jobs)
    assert all([r.data == script.main(**{k.lstrip("-"): v for k, v in j.args.items()}) for r, j in zip(results, jobs)])


@pytest.mark.timeout(10.0)
def test_local_from_fn():
    domain = ht.Domain({"x": [0., 1.]}, seed=7)
    jobs = [Job(task=square, args=(domain.sample(),)) for _ in range(10)]
    results = run_jobs(jobs)
    assert all([r.data.value == square(*j.args).value for r, j in zip(results, jobs)])


@pytest.mark.slurm
@pytest.mark.timeout(60.0)
def test_slurm_from_script():
    domain = ht.Domain({"x": {0, 1, 2, 3}, "y": [-1., 1.], "z": {"123", "abc"}}, seed=7)
    jobs, dirs = [], []
    n_jobs = 4
    for i in range(n_jobs):
        sample = domain.sample()
        # NOTE: this test might fail if /tmp is not shared in the slurm cluster.
        #  Adding the argument dir="/path/to/shared/dir" can fix that
        dirs.append(tempfile.TemporaryDirectory())
        jobs.append(SlurmJob(task="hypertunity/scheduling/tests/script.py",
                             args=(*sample.as_namedtuple(),),
                             output_file=f"{os.path.join(dirs[-1].name, 'results.pkl')}",
                             meta={"binary": "python", "resources": {"cpu": 1}}))
    results = run_jobs(jobs)
    assert all([r.data == script.main(*j.args) for r, j in zip(results, jobs)])
    # clean-up the temporary dirs
    for d in dirs:
        d.cleanup()
