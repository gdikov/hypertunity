"""Bayesian Optimisation using Gaussian Process regression."""

from multiprocessing import cpu_count
from typing import List, Dict, Tuple, TypeVar, Any, Sequence, Union, Type

import GPy
import GPyOpt
import numpy as np

from hypertunity import utils
from hypertunity.optimisation.base import Optimiser, EvaluationScore
from hypertunity.optimisation.domain import Domain, Sample

__all__ = [
    "BayesianOptimisation",
    "BayesianOptimization"
]

GPyOptSample = TypeVar("GPyOptSample", List[List], np.ndarray)
GPyOptDomain = List[Dict[str, Any]]
GPyOptCategoricalValueMapper = Dict[str, Dict[Any, int]]


class BayesianOptimisation(Optimiser):
    """Bayesian Optimiser using GPyOpt as a backend."""

    CONTINUOUS_TYPE = "continuous"
    DISCRETE_TYPE = "discrete"
    CATEGORICAL_TYPE = "categorical"

    def __init__(self, domain, seed=None):
        """Initialise the BO's domain and other options.

        Args:
            domain: `Domain` object defining the domain of the objective
            seed: optional, int to seed the BO for reproducibility.

        Notes:
             No objective function is specified for the `GPyOpt.methods.BayesianOptimisation` object as the evaluation
             is deferred to the user.
        """
        np.random.seed(seed)
        domain = Domain(domain.as_dict(), seed=seed)
        super(BayesianOptimisation, self).__init__(domain)
        self.gpyopt_domain, self._categorical_value_mapper = self._convert_to_gpyopt_domain(self.domain)
        self._inv_categorical_value_mapper = {name: {v: k for k, v in mapping.items()}
                                              for name, mapping in self._categorical_value_mapper.items()}
        self._data_x = np.array([[]])
        self._data_fx = np.array([[]])
        self.__is_empty_data = True

    @staticmethod
    def _convert_to_gpyopt_domain(orig_domain: Domain) -> Tuple[GPyOptDomain, GPyOptCategoricalValueMapper]:
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
            dim_name = utils.join_strings(names)
            domain_type = Domain.get_type(vals)
            if domain_type == Domain.Continuous:
                dim_type = BayesianOptimisation.CONTINUOUS_TYPE
            elif domain_type == Domain.Discrete:
                dim_type = BayesianOptimisation.DISCRETE_TYPE
            elif domain_type == Domain.Categorical:
                dim_type = BayesianOptimisation.CATEGORICAL_TYPE
                value_mapper[dim_name] = {v: i for i, v in enumerate(vals)}
                vals = tuple(range(len(vals)))
            else:
                raise ValueError(f"Badly specified subdomain {names} with values {vals}.")
            gpyopt_domain.append({"name": dim_name, "type": dim_type, "domain": tuple(vals)})
        assert len(gpyopt_domain) == len(orig_domain), "Mismatching dimensionality after domain conversion."
        return gpyopt_domain, value_mapper

    def _convert_to_gpyopt_sample(self, orig_sample: Sample) -> GPyOptSample:
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
            keys = utils.split_string(dim["name"])
            val = orig_sample[keys]
            if dim["type"] == BayesianOptimisation.CATEGORICAL_TYPE:
                val = self._categorical_value_mapper[dim["name"]][val]
            gpyopt_sample.append(val)
        return np.asarray(gpyopt_sample)

    def _convert_from_gpyopt_sample(self, gpyopt_sample: GPyOptSample) -> Sample:
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
            names = utils.split_string(dim["name"])
            sub_dim = orig_sample
            for name in names[:-1]:
                sub_dim[name] = {}
                sub_dim = sub_dim[name]
            if dim["type"] == BayesianOptimisation.CATEGORICAL_TYPE:
                sub_dim[names[-1]] = self._inv_categorical_value_mapper[dim["name"]][value]
            else:
                sub_dim[names[-1]] = value
        return Sample(orig_sample)

    @utils.support_american_spelling
    def run_step(self, batch_size: int = 1, minimise: bool = False, **kwargs) -> List[Sample]:
        """Run one step of Bayesian optimisation with a GP regression surrogate model.

        The first sample of the domain is chosen at random. Only after the model has been updated with at least one
        (data point, evaluation score)-pair the GPs are built and the acquisition function computed and optimised.

        Args:
            batch_size: int, the number of samples to suggest at once. If more than one,
                there is no guarantee for optimality.
            minimise: bool, whether the objective should be minimised
            **kwargs: optional keyword arguments which will be passed to the
                backend `GPyOpt.methods.BayesianOptimisation` optimiser.

        Keyword Args:
            model: str or `GPy.Model` object, the surrogate model used by the backend optimiser.
            kernel: `GPy.Kern` object, the kernel used by the model.
            variance: float, the variance of the objective function.


        Returns:
            A list of `batch_size`-many `Sample`s from the domain at which the objective should be evaluated next.
        """
        if self.__is_empty_data:
            next_samples = [self.domain.sample() for _ in range(batch_size)]
        else:
            assert len(self._data_x) > 0 and len(self._data_fx) > 0, "Cannot initialise a BO method from empty data."
            default_kwargs = {
                "num_cores": min(batch_size, cpu_count() - 1),
                "normalize_Y": True,
                "acquisition_type": "EI",
                "de_duplication": True,
                "model_type": "GP",
                "evaluator_type": "local_penalization" if batch_size > 1 else "sequential"
            }
            if "model" in kwargs:
                model = kwargs.pop("model")
                # NOTE: Remove this test for model type after the bug in GPyOpt is fixed:
                #  https://github.com/SheffieldML/GPyOpt/issues/183
                if isinstance(model, str) and model.lower() == "gp_mcmc" and batch_size > 1:
                    raise ValueError("GP_MCMC model cannot be used with a batch size > 1 due to a bug in "
                                     "GPyOpt: https://github.com/SheffieldML/GPyOpt/issues/183")
                kernel = kwargs.pop("kernel", None)
                variance = kwargs.pop("variance", None)
                default_kwargs["model"] = self._build_model(model, kernel, variance)
                if variance is not None and all(np.atleast_1d(np.isclose(variance, 0.0))):
                    default_kwargs["exact_feval"] = True
            default_kwargs = _overwrite_dict(default_kwargs, kwargs)

            # NOTE: as of GPyOpt 1.2.5 adding new data to an existing model is not yet possible,
            #  hence the object recreation. This behaviour might be changed in future versions.
            #  In this case the code should be refactored such that `bo` is initialised once and `update` takes
            #  care of the extension of the (X, Y) samples.
            bo = GPyOpt.methods.BayesianOptimization(
                f=None, domain=self.gpyopt_domain,
                maximize=not minimise,
                X=self._data_x,
                # NOTE: the following hack is necessary due to a bug in GPyOpt. The code should be updated
                #  once this gets fixed: https://github.com/SheffieldML/GPyOpt/issues/180
                Y=(-1 + 2 * minimise) * self._data_fx,
                initial_design_numdata=len(self._data_x),
                batch_size=batch_size,
                **default_kwargs)
            gpyopt_samples = bo.suggest_next_locations()
            next_samples = [self._convert_from_gpyopt_sample(s) for s in gpyopt_samples]
        return next_samples

    def _build_model(self, model: Union[str, Type[GPy.Model]] = "GP",
                     kernel: GPy.kern.Kern = None,
                     variance: float = None):
        """Build the surrogate model for the GPyOpt BayesianOptimisation.

        The default model is 'gp'. In case of a large number of already evaluated samples,
        a 'sparse_gp' is used to speed up computation.

        Args:
            model: str or `GPy.Model`, the GP regression model.
            kernel: `GPy.kern.Kern`, the kernel of the GP regression model.
            variance: float, the variance of the evaluations (used only if supported by the model).

        Returns:
            A `GPy.Model` instance.
        """
        if isinstance(model, GPy.Model):
            return model
        if isinstance(model, str):
            model = model.lower()
            if model == "gp":
                return GPyOpt.models.GPModel(kernel=kernel, noise_var=variance,
                                             sparse=len(self._data_x) > 25)
            if model == "gp_mcmc":
                return GPyOpt.models.GPModel_MCMC(kernel=kernel, noise_var=variance)
            raise ValueError(f"Unknown model {model}. When supplying a custom kernel or "
                             f"the variance of the objective function, the model has to be "
                             f"one from {{'GP', 'GP_MCMC'}}. Otherwise you should supply a "
                             f"custom `GPy.Model` instance.")
        raise TypeError("Argument `model` must be of type str or `GPy.Model`.")

    def update(self, x, fx, **kwargs):
        """Update the surrogate model with the domain sample `x` and the function evaluation `fx`.

        Args:
            x: `Sample`, one sample of the domain of the objective function.
            fx: either a float, an `EvaluationScore` or a dict, mapping metric names to `EvaluationScore`s
                of the objective at `x`.
            **kwargs: unused by this model.
        """
        super(BayesianOptimisation, self).update(x, fx)
        # both `converted_x` and `array_fx` must be 2dim arrays
        if isinstance(x, Sample):
            converted_x, array_fx = self._convert_evaluation_sample(x, fx)
        elif isinstance(x, Sequence) and isinstance(fx, Sequence) and len(x) == len(fx):
            # append each history point to the tracked history and convert to numpy arrays
            converted_x, array_fx = map(
                np.concatenate, zip(*[self._convert_evaluation_sample(i, j) for i, j in zip(x, fx)]))
        else:
            raise ValueError("Update values for `x` and `f(x)` must be either "
                             "`Sample` and an evaluation or a list thereof.")

        if self._data_x.size == 0:
            self._data_x = converted_x
            self._data_fx = array_fx
        else:
            self._data_x = np.concatenate([self._data_x, converted_x])
            self._data_fx = np.concatenate([self._data_fx, array_fx])
        self.__is_empty_data = False

    def _convert_evaluation_sample(self, x, fx):
        if isinstance(fx, (float, int)):
            array_fx = np.array([[fx]])
        elif isinstance(fx, EvaluationScore):
            array_fx = np.array([[fx.value]])
        elif isinstance(fx, Dict):
            if not len(fx) == 1:
                raise NotImplementedError("Currently only evaluations with a single metric are supported.")
            array_fx = np.array([[list(fx.values())[0].value]])
        else:
            raise TypeError("Cannot update history for one sample and multiple evaluations. "
                            "Use batched update instead and provide a list of samples "
                            "and a list of evaluation metrics.")
        converted_x = self._convert_to_gpyopt_sample(x).reshape(1, -1)
        return converted_x, array_fx

    def reset(self):
        """Reset the optimiser for a fresh start."""
        super(BayesianOptimisation, self).reset()
        self._data_x = np.array([])
        self._data_fx = np.array([])
        self.__is_empty_data = True


BayesianOptimization = BayesianOptimisation


def _overwrite_dict(old_dict, new_dict):
    updated_old = {}
    # copy the old dict
    for key, value in old_dict.items():
        updated_old[key] = value
    # overwrite the existing and add the new values
    for key, value in new_dict.items():
        updated_old[key] = value
    return updated_old
