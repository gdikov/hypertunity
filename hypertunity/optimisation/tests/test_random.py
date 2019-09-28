from hypertunity.domain import Domain
from hypertunity.optimisation import random

from . import _common as test_utils


def test_random_simple_continuous():
    domain = Domain({"x": [-1., 6.]})
    rs = random.RandomSearch(domain=domain, seed=7)
    test_utils.evaluate_continuous_1d(rs, batch_size=50, n_steps=2)


def test_random_simple_mixed():
    domain = Domain({"x": [-5., 6.], "y": {"sin", "sqr"}, "z": set(range(4))})
    rs = random.RandomSearch(domain=domain, seed=1)
    test_utils.evaluate_heterogeneous_3d(rs, batch_size=50, n_steps=25)


def test_update():
    domain = Domain({"x": [-5., 6.]})
    rs = random.RandomSearch(domain)
    rs.update([domain.sample() for _ in range(4)], list(range(4)))
    rs.update(domain.sample(), {"score": 23.0})
    rs.update(domain.sample(), 2.0)
    assert len(rs.history) == 6
    rs.reset()
    assert len(rs.history) == 0
