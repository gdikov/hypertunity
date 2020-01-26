import queue

import pytest

from .. import utils

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield


def test_support_american_spelling():

    @utils.support_american_spelling
    def gb_spelling_func(minimise, optimise, maximise):
        return minimise, optimise, maximise

    expected = (True, 1, None)
    assert gb_spelling_func(minimise=True, optimise=1, maximise=None) == expected
    assert gb_spelling_func(minimize=True, optimize=1, maximize=None) == expected


@pytest.mark.parametrize("test_input,expectation", [
    (("vxc", "", "", "___"), nullcontext()),
    (("_", "_", ""), nullcontext()),
    (("asd",), nullcontext()),
    (("asd", "dxcv"), nullcontext()),
    (("asd", "\\", "\n"), pytest.raises(ValueError))
])
def test_split_and_join_strings(test_input, expectation):
    with expectation:
        assert test_input == utils.split_string(
            utils.join_strings(test_input, join_char="_"),
            split_char="_"
        )


def test_drain_queue():
    q = queue.Queue(10)
    elems = list(range(10))
    for i in elems:
        q.put(i)
    items = utils.drain_queue(q)
    assert items == elems
    with pytest.raises(queue.Empty):
        q.get_nowait()
