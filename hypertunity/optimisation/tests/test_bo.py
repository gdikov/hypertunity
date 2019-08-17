# -*- coding: utf-8 -*-
import pytest

import numpy as np

import hypertunity as ht
from . import _common as test_utils


def test_bo_update_and_reset():
    domain = ht.Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    bo = ht.BayesianOptimisation(domain, minimise=True, batch_size=1)
    samples = bo.run_step()
    n_reps = 3
    for i in range(n_reps):
        bo.update(samples[0], ht.EvaluationScore(2. * i), )
    assert len(bo._data_x) == n_reps and len(bo._data_fx) == n_reps
    assert np.all(bo._data_x == np.tile(bo._convert_to_gpyopt_sample(samples[0]), (n_reps, 1)))
    assert np.all(bo._data_fx == 2. * np.arange(n_reps).reshape(n_reps, 1))
    bo.reset()
    assert len(bo.history) == 0


@pytest.mark.slow
def test_bo_simple_continuous():
    domain = ht.Domain({"x": [-1., 6.]})
    bo = ht.BayesianOptimization(
        domain=domain,
        minimise=False,
        batch_size=2,
        seed=7)
    test_utils.evaluate_simple_continuous(bo, n_steps=7)


@pytest.mark.slow
def test_bo_simple_mixed():
    domain = ht.Domain({"x": [-5., 6.], "y": ("sin", "sqr"), "z": tuple(range(4))})
    bo = ht.BayesianOptimization(
        domain=domain,
        batch_size=7,
        minimise=False,
        seed=7)
    test_utils.evaluate_simple_mixed(bo, n_steps=3)
