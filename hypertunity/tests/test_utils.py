import queue

import pytest

from .. import utils


def test_support_american_spelling():
    @utils.support_american_spelling
    def gb_spelling_func(minimise, optimise, maximise):
        return minimise, optimise, maximise
    expected = (True, 1, None)
    assert gb_spelling_func(minimise=True, optimise=1, maximise=None) == expected
    assert gb_spelling_func(minimize=True, optimize=1, maximize=None) == expected


def test_split_and_join_strings():
    strings = [
        ("vxc", "", "", "___"),
        ("_", "_", ""),
        ("asd",),
        ("asd", "dxcv")
    ]
    for s in strings:
        assert s == utils.split_string(
            utils.join_strings(s, join_char="_"),
            split_char="_"
        )
    with pytest.raises(ValueError):
        utils.join_strings(["asd", "\\", "\n"])


def test_drain_queue():
    q = queue.Queue(10)
    elems = list(range(10))
    for i in elems:
        q.put(i)
    items = utils.drain_queue(q)
    assert items == elems
    with pytest.raises(queue.Empty):
        q.get_nowait()
