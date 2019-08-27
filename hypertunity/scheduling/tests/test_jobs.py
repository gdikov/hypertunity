import pytest

from ..jobs import Job


def test_repeating_id():
    _ = Job(task=sum, args=(), id=-100)
    with pytest.raises(ValueError):
        _ = Job(task=max, args=(), id=-100)
    _ = Job(task=sum, args=(), id=-99)


def test_callable_job():
    for args in [(1, 2), (-4, 2), (131212, 123123123)]:
        job = Job(task=lambda x, y: x + y, args=args)
        result = job()
        assert result.data == sum(args)
