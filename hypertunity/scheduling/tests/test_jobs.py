# -*- coding: utf-8 -*-
import pytest

from .. import jobs
from ..jobs import Job, Result


def test_repeating_id():
    _ = Job(id=1, args=(), func=sum)
    with pytest.raises(ValueError):
        _ = Job(id=1, args=(), func=max)
    _ = Job(id=2, args=(), func=sum)
    jobs.reset_job_registry()


def test_callable_job():
    for i, args in enumerate([(1, 2), (-4, 2), (131212, 123123123)]):
        job = Job(id=i, args=args, func=lambda x, y: x + y)
        assert job() == Result(id=job.id, data=sum(args))
    jobs.reset_job_registry()
