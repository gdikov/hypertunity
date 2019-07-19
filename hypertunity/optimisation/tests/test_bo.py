# -*- coding: utf-8 -*-
import pytest

from hypertunity.optimisation.bo import BayesianOptimization
from hypertunity.optimisation.domain import Domain
from . import _common as test_utils


@pytest.mark.slow
def test_gpyopt_bo_simple_continuous():
    domain = Domain({"x": [-1., 6.]})
    bo = BayesianOptimization(
        backend="gpyopt",
        domain=domain,
        minimise=False,
        batch_size=2,
        seed=7)
    test_utils.evaluate_simple_continuous(bo, n_steps=7)


@pytest.mark.slow
def test_bo_simple_mixed():
    domain = Domain({"x": [-5., 6.], "y": ("sin", "sqr"), "z": tuple(range(4))})
    bo = BayesianOptimization(
        backend="gpyopt",
        domain=domain,
        batch_size=7,
        minimise=False,
        seed=7)
    test_utils.evaluate_simple_mixed(bo, n_steps=3)
