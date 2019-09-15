import GPy
import numpy as np
import pytest

import hypertunity as ht
from . import _common as test_utils


def test_bo_update_and_reset():
    domain = ht.Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    bo = ht.BayesianOptimisation(domain, seed=7)
    samples = bo.run_step(batch_size=1, minimise=False)
    n_reps = 3
    for i in range(n_reps):
        bo.update(samples[0], ht.EvaluationScore(2. * i), )
    assert len(bo._data_x) == n_reps and len(bo._data_fx) == n_reps
    assert np.all(bo._data_x == np.tile(bo._convert_to_gpyopt_sample(samples[0]), (n_reps, 1)))
    assert np.all(bo._data_fx == 2. * np.arange(n_reps).reshape(n_reps, 1))
    bo.reset()
    assert len(bo.history) == 0


def test_bo_set_history():
    n_samples = 10
    domain = ht.Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    history = [ht.HistoryPoint(domain.sample(), {"score": ht.EvaluationScore(float(i))})
               for i in range(n_samples)]
    bo = ht.BayesianOptimisation(domain, seed=7)
    bo.history = history
    assert bo.history == history
    assert len(bo._data_x) == len(bo._data_fx) == len(history)


@pytest.mark.slow
def test_bo_simple_continuous():
    domain = ht.Domain({"x": [-1., 6.]})
    bo = ht.BayesianOptimization(domain=domain, seed=7)
    test_utils.evaluate_continuous_1d(bo, batch_size=2, n_steps=7)


@pytest.mark.slow
def test_bo_simple_mixed():
    domain = ht.Domain({"x": [-5., 6.], "y": {"sin", "sqr"}, "z": set(range(4))})
    bo = ht.BayesianOptimization(domain=domain, seed=7)
    test_utils.evaluate_heterogeneous_3d(bo, batch_size=7, n_steps=3)


@pytest.mark.slow
def test_bo_custom_model():
    domain = ht.Domain({"x": [-2., 2.]})
    bo = ht.BayesianOptimisation(domain=domain, seed=7)
    kernel = GPy.kern.RBF(1) + GPy.kern.Bias(1)
    n_steps = 3
    batch_size = 3
    all_samples = []
    all_evaluations = []
    first_samples = bo.run_step(batch_size=batch_size, minimise=False)
    xs = np.atleast_2d([s["x"] for s in first_samples])
    ys = np.atleast_2d(test_utils.continuous_heteroscedastic_1d(np.array([s["x"] for s in first_samples])))
    for i in range(n_steps):
        custom_model = GPy.models.GPHeteroscedasticRegression(xs, ys, kernel)
        samples = bo.run_step(batch_size, minimise=False, model=custom_model)
        evaluations = test_utils.continuous_heteroscedastic_1d(np.array([s["x"] for s in samples]))
        bo.update(samples, [ht.EvaluationScore(ev) for ev in evaluations])
        xs = np.concatenate([xs, np.atleast_2d([s["x"] for s in samples])], axis=0)
        ys = np.concatenate([ys, np.atleast_2d(evaluations)], axis=0)
        # gather the samples and evaluations for later assessment
        all_samples.extend([s["x"] for s in samples])
        all_evaluations.extend(evaluations)
    best_eval_index = int(np.argmax(all_evaluations))
    best_sample = all_samples[best_eval_index]
    assert np.isclose(best_sample, test_utils.CONT_HETEROSCED_1D_ARGMAX, atol=1e-1)


@pytest.mark.skip("Due to a bug in GPyOpt using GP_MCMC model can not be tested yet.")
@pytest.mark.slow
def test_bo_gp_mcmc_model():
    domain = ht.Domain({"x": [-1., 6.]})
    bo = ht.BayesianOptimization(domain=domain, seed=7)
    test_utils.evaluate_continuous_1d(bo, batch_size=1, n_steps=7,
                                      model="GP_MCMC", evaluator_type="sequential")
