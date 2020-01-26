import pytest

from ..jobs import Job


def test_repeating_id():
    _ = Job(task=sum, args=(), id=-100)
    with pytest.raises(ValueError):
        _ = Job(task=max, args=(), id=-100)
    _ = Job(task=sum, args=(), id=-99)


def test_callable_job():
    job_args = (131212, 123123123)
    job = Job(task=lambda x, y: x + y, args=job_args)
    result = job()
    assert result.data == sum(job_args)
