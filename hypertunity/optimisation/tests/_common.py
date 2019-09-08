import numpy as np

from hypertunity.optimisation import EvaluationScore

CONT_1D_ARGMAX = 3.989333
CONT_1D_MAX = 5.958363


def continuous_1d(x):
    """Compute x * sin(2x) + 2 if x in [0, 5] else 0."""
    fx = np.atleast_1d(x * np.sin(2 * x) + 2)
    fx[np.logical_and(x < 0, x > 5)] = 0.
    return fx


CONT_HETEROSCED_1D_ARGMAX = 0.0
CONT_HETEROSCED_1D_MAX = 2.0


def continuous_heteroscedastic_1d(x):
    """Compute 0.2 * x^4 - x^2 + 2 + eps
    where eps ~ N(0, |0.2 * x| + 1e-7) and x in [-2., 2]
    """
    rng = np.random.RandomState(7)
    noise = rng.normal(0., 0.2 * np.abs(x) + 1e-7)
    fx = np.atleast_1d(0.2 * x**4 - x**2 + 2 + noise)
    fx[np.logical_and(x < -2., x > 2.)] = 0.
    return fx


HETEROGEN_3D_ARGMAX = (6.0, "sqr", 0)
HETEROGEN_3D_MAX = 36.0


def heterogeneous_3d(x, y, z):
    """Compute `continuous_1d` + z if y == 'sin', else return x**2 - 3 * z
    where x is continuous, y is categorical ("sin", "sqr"), z is discrete.

    Args:
        x: float or np.ndarray, continuous variable         [-5.0, 6.0]
        y: str, categorical variable                        ("sin", "sqr")
        z: float or int or np.ndarray, discrete variable    (0, 1, 2, 3)
    """
    if y == "sin":
        return (continuous_1d(x) + z)[0]
    elif y == "sqr" and z in [0, 1, 2, 3]:
        return x ** 2 - 3 * z
    else:
        raise ValueError("`y` can only be 'sin' or 'sqr' and z [0, 1, 2, 3].")


DISCRETE_3D_ARGMAX = (4, 5, "large")
DISCRETE_3D_MAX = 3.0


def discrete_3d(x, y, z):
    """Compute c * x * y where c = 0.1 if z == "small" else 0.15.

    `x` and `y` are discrete numerical values, z is categorical.

    Args:
        x: int, discrete variable                           (1, 2, 3, 4)
        y: int, discrete variable                           (-3, 2, 5)
        z: str, categorical variable                        ("small", "large")
    """
    if x not in {1, 2, 3, 4} and y not in {-3, 2, 5} and z not in {"small", "large"}:
        raise ValueError("Outside the allowed domain.")
    if z == "small":
        return 0.1 * x * y
    return 0.15 * x * y


def evaluate_continuous_1d(opt, batch_size, n_steps, **kwargs):
    all_samples = []
    all_evaluations = []
    for i in range(n_steps):
        samples = opt.run_step(batch_size, minimise=False, **kwargs)
        evaluations = continuous_1d(np.array([s["x"] for s in samples]))
        opt.update(samples, [EvaluationScore(ev) for ev in evaluations], )
        # gather the samples and evaluations for later assessment
        all_samples.extend([s["x"] for s in samples])
        all_evaluations.extend(evaluations)
    best_eval_index = int(np.argmax(all_evaluations))
    best_sample = all_samples[best_eval_index]
    best_eval = all_evaluations[best_eval_index]
    assert np.isclose(best_sample, CONT_1D_ARGMAX, atol=1e-1)
    assert np.isclose(best_eval, CONT_1D_MAX, atol=1e-1)


def evaluate_heterogeneous_3d(opt, batch_size, n_steps):
    all_samples = []
    all_evaluations = []
    for i in range(n_steps):
        samples = opt.run_step(batch_size, minimise=False)
        evaluations = [heterogeneous_3d(s["x"], s["y"], s["z"]) for s in samples]
        opt.update(samples, [EvaluationScore(ev) for ev in evaluations], )
        # gather the samples and evaluations for later assessment
        all_samples.extend([(s["x"], s["y"], s["z"]) for s in samples])
        all_evaluations.extend(evaluations)
    best_eval_index = int(np.argmax(all_evaluations))
    best_sample = all_samples[best_eval_index]
    best_eval = all_evaluations[best_eval_index]
    assert np.isclose(best_sample[0], HETEROGEN_3D_ARGMAX[0], atol=1.0)
    assert best_sample[1:] == HETEROGEN_3D_ARGMAX[1:]
    assert np.isclose(best_eval, HETEROGEN_3D_MAX, atol=1.0)


def evaluate_discrete_3d(opt, batch_size, n_steps):
    all_samples = []
    all_evaluations = []
    for i in range(n_steps):
        samples = opt.run_step(batch_size, minimise=False)
        evaluations = [discrete_3d(s["x"], s["y"], s["z"]) for s in samples]
        opt.update(samples, [EvaluationScore(ev) for ev in evaluations], )
        # gather the samples and evaluations for later assessment
        all_samples.extend([(s["x"], s["y"], s["z"]) for s in samples])
        all_evaluations.extend(evaluations)
    best_eval_index = int(np.argmax(all_evaluations))
    best_sample = all_samples[best_eval_index]
    best_eval = all_evaluations[best_eval_index]
    assert best_sample == DISCRETE_3D_ARGMAX
    assert best_eval == DISCRETE_3D_MAX
