# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple, TypeVar, Any, Iterable

import GPyOpt
import numpy as np

from multiprocessing import cpu_count

import hypertunity.optimisation as ht_opt

from hypertunity import utils


GPyOptSample = TypeVar("GPyOptSample", List[List], np.ndarray)
GPyOptDomain = List[Dict[str, Any]]
GPyOptCategoricalValueMapper = Dict[str, Dict[Any, int]]


class GPyOptBackend(ht_opt.BaseOptimiser):
    """Wrapper Bayesian Optimiser using GPyOpt as a backend."""
    CONTINUOUS_TYPE = "continuous"
    DISCRETE_TYPE = "discrete"
    CATEGORICAL_TYPE = "categorical"

    @utils.support_american_spelling
    def __init__(self, domain, minimise, batch_size=1, seed=None, **backend_kwargs):
        """Initialise the BO wrapper's domain and other options.

        Args:
            domain: `Domain` object defining the domain of the objective
            minimise: bool, whether the objective should be minimised
            batch_size: int, the number of samples to suggest at once. If more than one,
                there is no guarantee for optimality.
            seed: optional, int to seed the BO for reproducibility.

        Keyword Args:
            backend_kwargs: additional kwargs supplied to initialise a `GPyOpt.methods.BayesianOptimisation` object.

        Notes:
             No objective function is specified for the `GPyOpt.methods.BayesianOptimisation` object as the evaluation
             is deferred to the user.
        """
        super(GPyOptBackend, self).__init__(domain, minimise=minimise)
        self.gpyopt_domain, self._categorical_value_mapper = self._convert_to_gpyopt_domain(self.domain)
        self._minimise = minimise
        self._batch_size = batch_size
        self._num_cores = min(self._batch_size, cpu_count() - 1)
        if batch_size > 1:
            self._evaluation_type = "local_penalization"
        else:
            self._evaluation_type = "sequential"
        self._backend_kwargs = backend_kwargs
        self.__is_empty_data = True
        self._data_x = np.array([[]])
        self._data_fx = np.array([[]])
        np.random.seed(seed)

    @staticmethod
    def _convert_to_gpyopt_domain(orig_domain: ht_opt.Domain) -> Tuple[GPyOptDomain, GPyOptCategoricalValueMapper]:
        """Convert a `Domain` type object to `GPyOptDomain`.

        Args:
            orig_domain: `Domain` to convert.

        Returns:
            A tuple of the converted `GPyOptDomain` object and a value mapper to assign each categorical
            value to an integer (0, 1, 2, 3 ...). This is done to abstract away the type of the categorical domain
            from the GPyOpt internals and thus arbitrary types are supported.

        Notes:
            The categorical options must be hashable. This behaviour may change in the future.
        """
        gpyopt_domain = []
        value_mapper = {}
        flat_domain = orig_domain.flatten()
        for names, vals in flat_domain.items():
            dim_name = _clean_and_join(names)
            domain_type = ht_opt.Domain.get_type(vals)
            if domain_type == ht_opt.Domain.Continuous:
                dim_type = GPyOptBackend.CONTINUOUS_TYPE
            elif domain_type == ht_opt.Domain.Discrete:
                dim_type = GPyOptBackend.DISCRETE_TYPE
            elif domain_type == ht_opt.Domain.Categorical:
                dim_type = GPyOptBackend.CATEGORICAL_TYPE
                value_mapper[dim_name] = {v: i for i, v in enumerate(vals)}
            else:
                raise ValueError(f"Badly specified subdomain {names} with values {vals}.")
            gpyopt_domain.append({"name": dim_name, "type": dim_type, "domain": tuple(vals)})
        assert len(gpyopt_domain) == len(orig_domain), "Mismatching dimensionality after domain conversion."
        return gpyopt_domain, value_mapper

    def _convert_to_gpyopt_sample(self, orig_sample: ht_opt.Sample) -> GPyOptSample:
        """Convert a sample of type `Sample` to type `GPyOptSample` and vice versa.

        If the function is supplied with a `GPyOptSample` type object it calls the dedicated function
        `self._convert_from_gpyopt_sample`.

        Args:
            orig_sample: `Sample` type object to be converted.

        Returns:
            A `GPyOptSample` type object with the same values as `orig_sample`.
        """
        gpyopt_sample = []
        # iterate in the order of the GPyOpt domain names
        for dim in self.gpyopt_domain:
            keys = _revert_clean_and_join(dim["name"])
            val = orig_sample[keys]
            if dim["type"] == GPyOptBackend.CATEGORICAL_TYPE:
                val = self._categorical_value_mapper[dim["name"]][val]
            gpyopt_sample.append(val)
        return np.asarray(gpyopt_sample)

    def _convert_from_gpyopt_sample(self, gpyopt_sample: GPyOptSample) -> ht_opt.Sample:
        """Convert `GPyOptSample` type object to the corresponding `Sample` type.

        This is a registered function for the `self._convert_sample` function dispatcher.

        Args:
            gpyopt_sample: `GPyOptSample` object to be converted.

        Returns:
            A `Sample` type object with the same values as `gpyopt_sample`.
        """
        if len(self.gpyopt_domain) != len(gpyopt_sample):
            raise ValueError(f"Cannot convert sample with mismatching dimensionality. "
                             f"The original space has {len(self.domain)} dimensions and the "
                             f"sample {len(gpyopt_sample)} dimensions.")
        orig_sample = {}
        for dim, value in zip(self.gpyopt_domain, gpyopt_sample):
            names = _revert_clean_and_join(dim["name"])
            sub_dim = orig_sample
            for name in names[:-1]:
                sub_dim[name] = {}
                sub_dim = sub_dim[name]
            sub_dim[names[-1]] = value
        return ht_opt.Sample(orig_sample)

    def _build_model(self):
        """Build the surrogate model for the GPyOpt BayesianOptimisation.

        The default model is 'GP'. In case of a large number of already evaluated samples,
        a 'sparseGP' is used to speed up computation.
        """
        # TODO: improve with a custom generic kernel or task-specific one.
        #  Account for noise in observations (possibly heteroscedastic).
        if len(self._data_x) > 25:
            return "sparseGP"
        return "GP"

    def _build_acquisition(self):
        """Build the acquisition function."""
        return "EI"

    def run_step(self, *args, **kwargs):
        """Run one step of Bayesian optimisation with a GP regression surrogate model.

        The first sample of the domain is chosen at random. Only after the model has been updated with at least one
        (data point, evaluation score)-pair the GPs are built and the acquisition function computed and optimised.

        Returns:
            A `Sample` from the domain at which the objective should be evaluated next.
        """
        if self.__is_empty_data:
            next_samples = tuple([self.domain.sample() for _ in range(self._batch_size)])
        else:
            assert len(self._data_x) > 0 and len(self._data_fx) > 0, "Cannot initialise a BO method from empty data."
            # NOTE: as of GPyOpt 1.2.5 adding new data to an existing model is not yet possible,
            #  hence the object recreation. This behaviour might be changed in future versions.
            #  In this case the code should be refactored such that `bo` is initialised once and `update` takes
            #  care of the extension of the (X, Y) samples.
            bo = GPyOpt.methods.BayesianOptimization(
                f=None, domain=self.gpyopt_domain,
                maximize=not self._minimise,
                X=self._data_x,
                Y=(-1 + 2 * self._minimise) * self._data_fx,     # this hack is necessary due to a bug in GPyOpt
                initial_design_numdata=len(self._data_x),
                batch_size=self._batch_size,
                num_cores=self._num_cores,
                evaluator_type=self._evaluation_type,
                model_type=self._build_model(),
                acquisition_type=self._build_acquisition(),
                de_duplication=True,
                **self._backend_kwargs)
            gpyopt_samples = bo.suggest_next_locations()
            next_samples = tuple([self._convert_from_gpyopt_sample(s) for s in gpyopt_samples])
        if self._batch_size == 1:
            return next_samples[0]
        return next_samples

    def update(self, x, fx):
        """Update the surrogate model with the domain sample `x` and the function evaluation `fx`.

        Args:
            x: `Sample`, one sample of the domain of the objective function.
            fx: `EvaluationScore`, the evaluation score of the objective at `x`
        """
        # both `converted_x` and `array_fx` must be 2dim arrays
        if isinstance(x, ht_opt.Sample) and isinstance(fx, ht_opt.EvaluationScore):
            self.history.append(ht_opt.HistoryPoint(x, fx))
            converted_x = self._convert_to_gpyopt_sample(x).reshape(1, -1)
            array_fx = np.array([[fx.value]])
        elif isinstance(x, Iterable) and isinstance(fx, Iterable) and len(x) == len(fx):
            self.history.extend([ht_opt.HistoryPoint(sample=i, score=j) for i, j in zip(x, fx)])
            converted_x = np.array([self._convert_to_gpyopt_sample(s) for s in x])
            array_fx = np.array([f.value for f in fx]).reshape(-1, 1)
        else:
            raise ValueError("Update values for `x` and `f(x)` must be either "
                             "`Sample` and `EvaluationScore` or a list thereof.")

        if self._data_x.size == 0:
            self._data_x = converted_x
            self._data_fx = array_fx
        else:
            self._data_x = np.concatenate([self._data_x, converted_x])
            self._data_fx = np.concatenate([self._data_fx, array_fx])
        self.__is_empty_data = False

    def reset(self):
        """Reset the optimiser for a fresh start."""
        super(GPyOptBackend, self).reset()
        self._data_x = np.array([])
        self._data_fx = np.array([])
        self.__is_empty_data = True


def _clean_and_join(strings):
    """Join list of strings with an underscore.

    The strings must contain string.printable characters only, otherwise an exception is raised.
    If one of the strings has already an underscore, it will be replace by a null character.

    Args:
        strings: iterable of strings

    Returns:
        The joined string with an underscore character.

    Examples:
    ```python
        >>> _clean_and_join(['asd', '', '_xcv__'])
        'asd__\x00\x00xcv\x00'
    ```

    Raises:
        ValueError if a string contains an unprintable character.
    """
    all_cleaned = []
    for s in strings:
        if not s.isprintable():
            raise ValueError("Encountered unexpected name containing non-printable characters.")
        all_cleaned.append(s.replace("_", "\0"))
    return "_".join(all_cleaned)


def _revert_clean_and_join(joined):
    """Split joined string and substitute back the null characters with an underscore if necessary.

    Inverse function of `_clean_and_join(strings)`.

    Args:
        joined: str, the joined representation of the substrings.

    Returns:
        A tuple of strings with the splitting character (underscore) removed.

    Examples:
    ```python
        >>> _revert_clean_and_join('asd__\x00\x00xcv\x00')
        ('asd', '', '_xcv__')
    ```
    """
    strings = joined.split("_")
    strings_copy = []
    for s in strings:
        strings_copy.append(s.replace("\0", "_"))
    return tuple(strings_copy)
