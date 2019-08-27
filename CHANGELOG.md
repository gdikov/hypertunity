# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2019-08-28
## Added
- `Scheduler` to run jobs locally using joblib.
- `SlurmJob` and `Job` dataclasses defining the tasks to be scheduled.
- `Result` dataclass encapsulating the results from the tasks.
- `TableReporter` class for reporting results in tabular format.
- `Reporter` base class for extending reporters.

## Changed
- `Base`-prefix is removed from all base classes which reside 
 in `base.py` modules.
- `split_by_type` is now a method of the `Domain` class.
- `Optimiser` has a `batch_size` attribute accessible as a property.

## Removed
- `optimisation.bo` package has been removed. Instead a single `bo.py`
 module supports the only BO backend---GPyOpt, as of now.
- prefix for the file encoding (default is utf-8).
 
## [0.1.0] - 2019-07-27
### Added
- `TensorboardReporter` result logger using `HParams`
- `GpyOpt` backend for `BayesianOptimisation`
- `RandomSearch` optimiser
- `GridSearch` optimiser
- `Domain` and `Sample` classes as foundations for the optimisers
