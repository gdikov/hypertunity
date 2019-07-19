# -*- coding: utf-8 -*-

import numpy as np

from hypertunity.optimisation.base import EvaluationScore
from hypertunity.optimisation.domain import Domain
from ..gpyopt import GPyOptBackend


def test_gpyopt_bo_update():
    domain = Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    bo = GPyOptBackend(domain, minimise=True, batch_size=1)
    samples = bo.run_step()
    n_reps = 3
    for i in range(n_reps):
        bo.update(samples[0], EvaluationScore(2. * i), )
    assert len(bo._data_x) == n_reps and len(bo._data_fx) == n_reps
    assert np.all(bo._data_x == np.tile(bo._convert_to_gpyopt_sample(samples[0]), (n_reps, 1)))
    assert np.all(bo._data_fx == 2. * np.arange(n_reps).reshape(n_reps, 1))


def test_gpyopt_bo_reset():
    domain = Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    bo = GPyOptBackend(domain, minimise=True, batch_size=1)
    samples = bo.run_step()
    n_reps = 3
    for i in range(n_reps):
        bo.update(samples[0], EvaluationScore(2. * i), )
    bo.reset()
    assert len(bo.history) == 0
