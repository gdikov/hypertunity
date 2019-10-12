Optimisation
============

Hypertunity ships with three types of hyperparameter space exploration algorithms. A Bayesian optimisation, random and
grid search. While the first one is sequential in nature and requires evaluations to update its internal model of the
objective function, so that more informed sample suggestions are generated, the latter two are able to generate all samples
in parallel and do not require updating. In this section we will give a brief overview of each.

Bayesian optimisation
---------------------

:class:`BayesianOptimisation` in Hypertunity is a wrapper around `GPyOpt.methods.BayesianOptimization` which uses
Gaussian Process regression to build a surrogate model of the objective function. It is initialised from a :class:`Domain`
object:

.. code-block:: python

    bo = BayesianOptimization(domain)

The :class:`BayesianOptimisation` optimiser is highly customisable during sampling. This enables the user to
dynamically refine the model during calling :py:meth:`run_step()`. This approach introduces however the computational
burden of recomputing the surrogate model at each query. In the following example we show how one can set the GP model
using readily available ones from `GPy.models`, e.g. a `GPHeteroschedasticRegression`:

.. code-block:: python

    bo = BayesianOptimisation(domain=domain, seed=7)                    # initialise BO optimiser
    kernel = GPy.kern.RBF(1) + GPy.kern.Bias(1)                         # create a custom kernel
    custom_model = GPy.models.GPHeteroscedasticRegression(..., kernel)  # create a custom model
    samples = bayes_opt.run_step(model=custom_model)                    # generate samples


Random search
-------------

This class is a wrapper around the :py:meth:`Domain.sample()` method. It has the API of
an :class:`Optimiser` class and yields samples which are uniformly drawn from the domain.
There is no limitation on the number of samples that can be returned in a single call of :py:meth:`run_step()`,
even if this leads to repetitions.


Grid search
-----------

:class:`GridSearch` is a wrapper around the iteration over a domain. It goes over each point in the Cartesian-product of
all discrete subdomains. If one of the subdomains is continuous :class:`GridSearch` will sample uniformly from
this interval. Once the domain is exhausted, further iteration will be prevented by raising an :class:`ExhaustedSearchSpaceError`.
To iterate again the :class:`GridSearch` optimiser must be reset by calling the :py:meth:`reset()` method.

.. code-block:: python

    >>> domain = Domain({"x": {1, 2, 3}, "y": {"a", "b"}, "z": [0, 1]})
    >>> gs = GridSearch(domain, sample_continuous=True)
    >>> gs.run_step(batch_size=6)
    [
        {'x': 1, 'y': 'b', 'z': 0.054781406913364084},
        {'x': 2, 'y': 'b', 'z': 0.7006391867439882},
        {'x': 3, 'y': 'b', 'z': 0.9674445624792569},
        {'x': 1, 'y': 'a', 'z': 0.7837727333178091},
        {'x': 2, 'y': 'a', 'z': 0.17240297136803384},
        {'x': 3, 'y': 'a', 'z': 0.844465575155033}
    ]
    >>> gs.reset()




Custom optimiser
----------------

If neither of the predefined optimiser are useful for your problem, you can easily roll out a custom one.
Only thing you have to do is to inherit from the base :class:`Optimiser` class and implement the :py:meth:`run_step` method.

.. code-block:: python

    class CustomOptimiser(Optimiser):
        def __init__(self, domain, *args, **kwargs):
            super(CustomOptimiser, self).__init__(domain)
            ...

        def run_step(batch_size, *args, **kwargs):
            ...
            return [samples]
