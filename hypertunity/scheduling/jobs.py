# -*- coding: utf-8 -*-
"""Definition of `Job` and `Result` classes used to encapsulate an experiment and the corresponding outcomes."""

from dataclasses import dataclass
from typing import Callable, Tuple, Any

JOB_REGISTRY = set()


def reset_job_registry():
    """Reset the global job registry.

    Notes:
        **It is intended to be used during testing only.**

        This function should be used with care as it allows for jobs with repeating IDs to be created.
        As a consequence, two or more `Result` objects might coexist end make the actual experiment outcome
        ambiguous.


    """
    JOB_REGISTRY.clear()


@dataclass(frozen=True)
class Job:
    """`Job` class defining an experiment as a runnable task.

    Attributes:
        id: int, the job identifier. Must be unique.
        args: tuple of arguments for the callable function or script.
        func: callable, the python function to run on the args which will produce a `Result` object.
    """
    id: int
    args: Tuple
    func: Callable = None

    def __post_init__(self):
        if self.id in JOB_REGISTRY:
            raise ValueError(f"Job with id {self.id} is already created and cannot be overwritten.")
        JOB_REGISTRY.add(self.id)

    def __hash__(self):
        return hash(str(self.id))

    def __call__(self, *args, **kwargs) -> 'Result':
        all_args = args + self.args
        return Result(id=self.id, data=self.func(*all_args, **kwargs))


@dataclass(frozen=True)
class Result:
    """`Result` of the executed `Job` sharing the same id as the job.

    Attributes:
        id: int, the identifier of the `Result` object which corresponds to the job that has been run.
        data: Any, the outcome of the experiment.
    """
    id: int
    data: Any
