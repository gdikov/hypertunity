import pytest

from hypertunity.optimisation import domain as opt
from hypertunity.optimisation.exhaustive import GridSearch, ExhaustedSearchSpaceError
from . import _common as test_utils


def test_grid_simple_discrete():
    domain = opt.Domain({"x": {1, 2, 3, 4}, "y": {-3, 2, 5}, "z": {"small", "large"}})
    gs = GridSearch(domain=domain)
    test_utils.evaluate_simple_discrete(gs, batch_size=4, n_steps=3 * 2)
    with pytest.raises(ExhaustedSearchSpaceError):
        gs.run_step(batch_size=4)
    gs.reset()
    assert len(gs.run_step(batch_size=4)) == 4


def test_grid_simple_mixed():
    domain = opt.Domain({"x": [-5., 6.], "y": {"sin", "sqr"}, "z": set(range(4))})
    with pytest.raises(opt.DomainNotIterableError):
        _ = GridSearch(domain)
    gs = GridSearch(domain, sample_continuous=True, seed=93)
    assert len(gs.run_step(batch_size=8)) == 8
