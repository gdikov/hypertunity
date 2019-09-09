import pytest

from hypertunity.optimisation import domain as opt
from hypertunity.optimisation.exhaustive import GridSearch, ExhaustedSearchSpaceError
from . import _common as test_utils


def test_grid_simple_discrete():
    domain = opt.Domain({"x": {1, 2, 3, 4}, "y": {-3, 2, 5}, "z": {"small", "large"}})
    gs = GridSearch(domain=domain)
    test_utils.evaluate_discrete_3d(gs, batch_size=4, n_steps=3 * 2)
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


def test_update():
    domain = opt.Domain({"x": {-5., 6.}})
    gs = GridSearch(domain)
    gs.update([domain.sample() for _ in range(10)], list(range(10)))
    gs.update(domain.sample(), {"score": 23.0})
    gs.update(domain.sample(), 2.0)
    assert len(gs.history) == 12
