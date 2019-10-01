import GPy
import numpy as np
import pytest

from hypertunity.domain import Domain
from hypertunity.optimisation import base
from hypertunity.optimisation import bo

from . import _common as test_utils


def test_bo_update_and_reset():
    domain = Domain({"a": {"b": [2, 3], "d": {"f": [3, 4]}}, "c": [0, 0.1]})
    bayes_opt = bo.BayesianOptimisation(domain, seed=7)
    samples = []
    n_reps = 3
    for i in range(n_reps):
        samples.extend(bayes_opt.run_step(batch_size=1, minimise=False))
        bayes_opt.update(samples[-1], base.EvaluationScore(2. * i))
    assert len(bayes_opt._data_x) == n_reps and len(bayes_opt._data_fx) == n_reps
    assert np.all(bayes_opt._data_x == np.array([bayes_opt._convert_to_gpyopt_sample(s) for s in samples]))
    assert np.all(bayes_opt._data_fx == 2. * np.arange(n_reps).reshape(n_reps, 1))
    bayes_opt.reset()
    assert len(bayes_opt.history) == 0


def test_bo_set_history():
    n_samples = 10
    domain = Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    history = [base.HistoryPoint(domain.sample(), {"score": base.EvaluationScore(float(i))})
               for i in range(n_samples)]
    bayes_opt = bo.BayesianOptimisation(domain, seed=7)
    bayes_opt.history = history
    assert bayes_opt.history == history
    assert len(bayes_opt._data_x) == len(bayes_opt._data_fx) == len(history)


@pytest.mark.slow
def test_bo_simple_continuous():
    domain = Domain({"x": [-1., 6.]})
    bayes_opt = bo.BayesianOptimization(domain=domain, seed=7)
    test_utils.evaluate_continuous_1d(bayes_opt, batch_size=2, n_steps=7)


@pytest.mark.slow
def test_bo_simple_mixed():
    domain = Domain({"x": [-5., 6.], "y": {"sin", "sqr"}, "z": set(range(4))})
    bayes_opt = bo.BayesianOptimization(domain=domain, seed=7)
    test_utils.evaluate_heterogeneous_3d(bayes_opt, batch_size=7, n_steps=3)


@pytest.mark.slow
def test_bo_custom_model():
    domain = Domain({"x": [-2., 2.]})
    bayes_opt = bo.BayesianOptimisation(domain=domain, seed=7)
    kernel = GPy.kern.RBF(1) + GPy.kern.Bias(1)
    n_steps = 3
    batch_size = 3
    all_samples = []
    all_evaluations = []
    first_samples = bayes_opt.run_step(batch_size=batch_size, minimise=False)
    xs = np.atleast_2d([s["x"] for s in first_samples])
    ys = np.atleast_2d(test_utils.continuous_heteroscedastic_1d(np.array([s["x"] for s in first_samples])))
    for i in range(n_steps):
        custom_model = GPy.models.GPHeteroscedasticRegression(xs, ys, kernel)
        samples = bayes_opt.run_step(batch_size, minimise=False, model=custom_model)
        evaluations = test_utils.continuous_heteroscedastic_1d(np.array([s["x"] for s in samples]))
        bayes_opt.update(samples, [base.EvaluationScore(ev) for ev in evaluations])
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
    domain = Domain({"x": [-1., 6.]})
    bayes_opt = bo.BayesianOptimization(domain=domain, seed=7)
    test_utils.evaluate_continuous_1d(bayes_opt, batch_size=1, n_steps=7,
                                      model="GP_MCMC", evaluator_type="sequential")
