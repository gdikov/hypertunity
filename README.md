<div align="center">
  <img src="https://raw.githubusercontent.com/gdikov/hypertunity/master/docs/_static/images/logo.svg?sanitize=true" width="100%">
</div>

[![CircleCI](https://img.shields.io/circleci/build/github/gdikov/hypertunity)](https://circleci.com/gh/gdikov/hypertunity)
[![Documentation Status](https://readthedocs.org/projects/hypertunity/badge/?version=latest)](https://hypertunity.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/gdikov/hypertunity)

## Why Hypertunity

Hypertunity is a lightweight, high-level library for hyperparameter optimisation. 
Among others, it supports:
 * Bayesian optimisation by wrapping [GPyOpt](http://sheffieldml.github.io/GPyOpt/),
 * external or internal objective function evaluation by a scheduler, also compatible with [Slurm](https://slurm.schedmd.com),
 * real-time visualisation of results in [Tensorboard](https://www.tensorflow.org/tensorboard) 
 via the [HParams](https://www.tensorflow.org/tensorboard/r2/hyperparameter_tuning_with_hparams) plugin.

For the full set of features refer to the [documentation](https://hypertunity.readthedocs.io).

## Quick start

Define the objective function to optimise. For example, it can take the hyperparameters `params` as input and 
return a raw value `score` as output:

```python
import hypertunity as ht

def foo(**params) -> float:
    # do some very costly computations
    ...
    return score
```

To define the valid ranges for the values of `params` we create a `Domain` object:

```python
domain = ht.Domain({
    "x": [-10., 10.],         # continuous variable within the interval [-10., 10.]
    "y": {"opt1", "opt2"},    # categorical variable from the set {"opt1", "opt2"}
    "z": set(range(4))        # discrete variable from the set {0, 1, 2, 3}
})
```

Then we set up the optimiser:

```python
bo = ht.BayesianOptimisation(domain=domain)
```

And we run the optimisation for 10 steps. Each result is used to update the optimiser so that informed domain 
samples are drawn:

```python
n_steps = 10
for i in range(n_steps):
    samples = bo.run_step(batch_size=2, minimise=True)      # suggest next samples
    evaluations = [foo(**s.as_dict()) for s in samples]     # evaluate foo
    bo.update(samples, evaluations)                         # update the optimiser
```

Finally, we visualise the results in Tensorboard: 

```python
import hypertunity.reports.tensorboard as tb

results = tb.Tensorboard(domain=domain, metrics=["score"], logdir="path/to/logdir")
results.from_history(bo.history)
```

## Even quicker start

A high-level wrapper class `Trial` allows for seamless parallel optimisation
without bothering with scheduling jobs, updating optimisers and logging:
   
```python
trial = ht.Trial(objective=foo,
                 domain=domain,
                 optimiser="bo",
                 reporter="tensorboard",
                 metrics=["score"])
trial.run(n_steps, batch_size=2, n_parallel=2)
```

## Installation

### Using PyPI
To install the base version run:
```bash
pip install hypertunity
```
To use the Tensorboard dashboard, build the docs or run the test suite you will need the following extras:
```bash
pip install hypertunity[tensorboard,docs,tests]
```

### From source
Checkout the latest master and install locally:
```bash
git clone https://github.com/gdikov/hypertunity.git
cd hypertunity
pip install ./[tensorboard]
```
