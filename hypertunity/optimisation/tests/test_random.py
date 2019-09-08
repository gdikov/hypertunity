from hypertunity.optimisation.domain import Domain
from hypertunity.optimisation.random import RandomSearch
from . import _common as test_utils


def test_random_simple_continuous():
    domain = Domain({"x": [-1., 6.]})
    rs = RandomSearch(
        domain=domain,
        seed=7)
    test_utils.evaluate_continuous_1d(rs, batch_size=50, n_steps=2)


def test_random_simple_mixed():
    domain = Domain({"x": [-5., 6.], "y": {"sin", "sqr"}, "z": set(range(4))})
    rs = RandomSearch(
        domain=domain,
        seed=1)
    test_utils.evaluate_heterogeneous_3d(rs, batch_size=50, n_steps=25)
