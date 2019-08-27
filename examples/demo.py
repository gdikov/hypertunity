import numpy as np

import hypertunity as ht
import hypertunity.reports.tensorboard as tb


def foo(x, y, z):
    """Compute `bar` + z if y == 'sin', else return x**2 - 3 * z

    Args:
        x: float, continuous variable   [-5.0, 6.0]
        y: str, categorical variable    ("sin", "sqr")
        z: int, discrete variable       (0, 1, 2, 3)
    """

    def bar(x):
        """Compute x * sin(2x) + 2 if x in [0, 5] else 0."""
        fx = np.atleast_1d(x * np.sin(2 * x) + 2)
        fx[np.logical_and(x < 0, x > 5)] = 0.
        return fx

    if y == "sin":
        return (bar(x) + z)[0]
    elif y == "sqr" and z in [0, 1, 2, 3]:
        return x**2 - 3 * z
    else:
        raise ValueError("`y` can only be 'sin' or 'sqr' and z [0, 1, 2, 3].")


if __name__ == '__main__':
    domain = ht.Domain({"x": [-5., 6.], "y": ("sin", "sqr"), "z": tuple(range(4))})
    bo = ht.BayesianOptimization(
        domain=domain,
        minimise=False,
        batch_size=2)

    n_steps = 6
    with ht.Scheduler(n_parallel=bo.batch_size) as scheduler:
        for i in range(n_steps):
            samples = bo.run_step()
            jobs = [ht.Job(task=foo, args=(*s.as_namedtuple(),)) for s in samples]
            scheduler.dispatch(jobs)
            evaluations = [r.data for r in scheduler.collect(n_results=bo.batch_size, timeout=10.0)]
            bo.update(samples, [ht.EvaluationScore(ev) for ev in evaluations])

    rep = ht.TableReporter(domain, metrics=["score"])
    rep.from_history(bo.history)
    print(rep.format(order="none", emphasise=True))
    tb_rep = tb.TensorboardReporter(domain, metrics=["score"], logdir="./logs")
    tb_rep.from_history(bo.history)
