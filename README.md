# Hypertunity
A toolset for black-box hyperparameter optimisation.

[![CircleCI](https://circleci.com/gh/gdikov/hypertunity/tree/master.svg?style=svg&circle-token=1e875efacfef7d74c4ae07321d6be6d8482a13b1)](https://circleci.com/gh/gdikov/hypertunity/tree/master)

## Why Hypertunity

_Hypertunity_ is a lightweight, high-level library for hyperparameter optimisation. 
Among others, it supports:
 * Bayesian optimisation by wrapping [GPyOpt](http://sheffieldml.github.io/GPyOpt/),
 * external or internal objective function evaluation by a scheduler, also compatible with [Slurm](https://slurm.schedmd.com),
 * real-time visualisation of results in [Tensorboard](https://www.tensorflow.org/tensorboard) 
 via the [HParams](https://www.tensorflow.org/tensorboard/r2/hyperparameter_tuning_with_hparams) plugin.

For the full set of features refer to the [documentation](tbd).

## Quick start

A central object in _Hypertunity_ is the `Domain` defining the space of valid values for an objective function.
A `Sample` is a one realisation from the `Domain`, which supplied to the objective function results in an
`EvaluationScore`——a numeric value representing the goodness of the sample.

Define a wrapper around an expensive objective function that takes a `Sample` and returns an `EvaluationScore`:
```python
import hypertunity as ht

def foo(x: ht.Sample) -> ht.EvaluationScore:
    # do some very costly computations
    ...
    return ht.EvaluationScore(score, variance)
```
Define the valid ranges of values for `foo` and the optimiser:

```python
# define the optimisation domain
domain = ht.Domain({"x": [-5., 6.],           # continuous variable within the interval [-5., 6.]
                    "y": {"opt1", "opt2"},    # categorical variable from the set {"opt1", "opt2"}
                    "z": set(range(4))})    # discrete variable from the set {0, 1, 2, 3}

# initialise a BO optimiser
bo = ht.BayesianOptimisation(domain=domain)
```

Run the optimisation for 10 steps, while updating the optimiser:

```python
n_steps = 10
for i in range(n_steps):
    # suggest next samples from the domain
    samples = bo.run_step(batch_size=2, minimise=True)
    # evaluate the costly objective `foo`
    evaluations = [foo(s) for s in samples]
    # update the optimiser with the results
    bo.update(samples, evaluations)
```

Finally, visualise the results in Tensorboard: 

```python
import hypertunity.reports.tensorboard as tb

results = tb.TensorboardReporter(domain=domain, metrics=["score"], logdir="path/to/logdir")
results.from_history(bo.history)
```

## Installation

Checkout the latest master and install from source:
```bash
git clone https://github.com/gdikov/hypertunity.git
cd hypertunity
pip install ./[tensorboard]
```