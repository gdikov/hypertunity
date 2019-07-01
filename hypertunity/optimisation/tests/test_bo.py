# -*- coding: utf-8 -*-

import numpy as np

from hypertunity.optimisation.domain import Domain
from hypertunity.optimisation.base import EvaluationScore
from ..bo import BayesianOptimization


_SIMPLE_FUNC_ARGMAX = 3.989333
_SIMPLE_FUNC_MAX = 5.958363


def simple_func(x):
    """Compute x * sin(2x) + 2 if x in [0, 5] else 0."""
    fx = np.atleast_1d(x * np.sin(2 * x) + 2)
    fx[np.logical_and(x < 0, x > 5)] = 0.
    return fx


def test_gpyopt_bo_simple_continuous():
    domain = Domain({"x": [-1., 6.]}, seed=7)
    bo = BayesianOptimization(
        backend="gpyopt",
        domain=domain,
        minimise=False,
        batch_size=2,
        seed=7)
    all_samples = []
    all_evaluations = []
    n_steps = 7
    for i in range(n_steps):
        samples = bo.run_step()
        evaluations = simple_func(np.array([s["x"] for s in samples]))
        bo.update(samples, [EvaluationScore(ev) for ev in evaluations])
        # gather the samples and evaluations for later assessment
        all_samples.extend([s["x"] for s in samples])
        all_evaluations.extend(evaluations)
    best_eval_index = np.argmax(all_evaluations)
    best_sample = all_samples[best_eval_index]
    best_eval = all_evaluations[best_eval_index]
    assert np.isclose(best_sample, _SIMPLE_FUNC_ARGMAX, atol=1e-1)
    assert np.isclose(best_eval, _SIMPLE_FUNC_MAX, atol=1e-1)
