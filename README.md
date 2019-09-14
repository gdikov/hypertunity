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

A central object is the `Domain` defining the space of valid values for an objective function.
A `Sample` is a one realisation from the `Domain`, which supplied to the objective function results in a score.

Define a wrapper around an expensive objective function:
```python
import hypertunity as ht

def foo(x: ht.Sample) -> float:
    # do some very costly computations
    ...
    return score
```
Define the valid ranges of values for `foo`:

```python
# define the optimisation domain
domain = ht.Domain({"x": [-5., 6.],           # continuous variable within the interval [-5., 6.]
                    "y": {"opt1", "opt2"},    # categorical variable from the set {"opt1", "opt2"}
                    "z": set(range(4))})      # discrete variable from the set {0, 1, 2, 3}
```

Set up the optimiser:

```python
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

## Even quicker start

A high-level wrapper class `Trial` allows for seamless parallel optimisation
without bothering with scheduling jobs, updating optimisers and logging:
   
```python
trial = ht.Trial(objective=foo, domain=domain, optimiser="bo", 
                 reporter="tensorboard", metrics=["score"])
trial.run(n_steps, batch_size=2, n_parallel=2)
```

## Installation

Checkout the latest master and install from source:
```bash
git clone https://github.com/gdikov/hypertunity.git
cd hypertunity
pip install ./[tensorboard]
```