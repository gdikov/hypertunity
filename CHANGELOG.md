# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
## Added
- `SlurmScheduler` to perform distributed evaluation with slurm
- `LocalScheduler` to perform locally evaluations using joblib

## Fixed
- Typos in the documentation 

## [0.1] - 2019-07-26
### Added
- `TensorboardReporter` result logger using `HParams`
- `GpyOpt` backend for `BayesianOptimisation`
- `RandomSearch` optimiser
- `GridSearch` optimiser
- `Domain` and `Sample` classes as foundations for the optimisers
