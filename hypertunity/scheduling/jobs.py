# -*- coding: utf-8 -*-
"""Definition of `Job` and `Result` classes used to encapsulate an experiment and the corresponding outcomes."""

from dataclasses import dataclass, field
from typing import Callable, Tuple, Any, Union

from hypertunity.utils import import_script


_JOB_REGISTRY = set()
_RESULT_REGISTRY = set()
_ID_COUNTER = -1


def reset_registry():
    """Reset the global job and result registries.

    Notes:
        **It is intended to be used during testing only.**

        This function should be used with care as it allows for jobs with repeating IDs to be created.
        As a consequence, two or more `Result` objects might coexist end make the actual experiment outcome
        ambiguous.
    """
    global _ID_COUNTER
    _JOB_REGISTRY.clear()
    _RESULT_REGISTRY.clear()
    _ID_COUNTER = -1


def generate_id():
    """Generate a new, unused integer job id."""
    global _ID_COUNTER
    _ID_COUNTER += 1
    return _ID_COUNTER


def script_to_func(script_path: str) -> Callable:
    """Convert a module to a callable function and call the `main` function of the module.

    Args:
        script_path: str, the file path to the python script to run. It can either be given as a module
            i.e. in the [package.]*[module] form, or as a path to a *.py file in case it is not added
            into the PYTHONPATH environment variable.

    Returns:
        The wrapper which calls the main function from the script module.

    Raises:
          `AttributeError` if the script does not define a `main` function.
    """
    def wrapper(*args):
        module = import_script(script_path)
        if not hasattr(module, "main"):
            raise AttributeError(f"Cannot find 'main' function in {script_path}.")
        return module.main(*args)
    return wrapper


@dataclass(frozen=True)
class Job:
    """`Job` class defining an experiment as a runnable task.

    The job is defined by a callable function or a script task. In the case of the former the `args` will be passed
    directly to it upon calling. Otherwise a `main` attribute will be run with the `args`. In both cases a
    `Result` object should be returned.

    Attributes:
        id: int, the job identifier. Must be unique.
        args: tuple of arguments for the callable function or script.
        task: callable or str, a python function to run or a file path to a python script.
    """
    task: Union[Callable, str]
    args: Tuple = ()
    id: int = field(default_factory=generate_id)

    def __post_init__(self):
        if not isinstance(self.task, (Callable, str)):
            raise ValueError("Job's task must be either a callable function or a path to a script.")
        if self.id in _JOB_REGISTRY:
            raise ValueError(f"Job with id {self.id} is already created. Reusing ids is prohibited.")
        _JOB_REGISTRY.add(self.id)

    def __hash__(self):
        return hash(str(self.id))

    def __call__(self, *args, **kwargs) -> 'Result':
        all_args = args + self.args
        if isinstance(self.task, Callable):
            runnable = self.task
        else:
            runnable = script_to_func(self.task)
        return Result(id=self.id, data=runnable(*all_args, **kwargs))


@dataclass(frozen=True)
class Result:
    """`Result` of the executed `Job` sharing the same id as the job.

    Attributes:
        id: int, the identifier of the `Result` object which corresponds to the job that has been run.
        data: Any, the outcome of the experiment.
    """
    data: Any
    id: int

    def __post_init__(self):
        if self.id in _RESULT_REGISTRY:
            raise ValueError(f"Result with id {self.id} is already created. Reusing ids is prohibited.")
        _RESULT_REGISTRY.add(self.id)
