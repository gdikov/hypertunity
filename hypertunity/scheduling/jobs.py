"""Definition of `Job` and `Result` classes used to encapsulate an experiment and the corresponding outcomes."""

import enum
import importlib
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Any, Union

__all__ = [
    "Job",
    "SlurmJob",
    "Result"
]

# Global registries to control the job and result id assignment
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


def import_script(path):
    """Import a module or script by a given path.

    Args:
        path: str, can be either a module import of the form [package.]*[module]
            if the outer most package is in the PYTHONPATH, or a path to an arbitrary python script.

    Returns:
        The loaded python script as a module.
    """
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot find script {path}.")
        if not os.path.basename(path).endswith(".py"):
            raise ValueError(f"Expected a python script ending with *.py, found {os.path.basename(path)}.")
        import_path = os.path.dirname(os.path.abspath(path))
        sys.path.append(import_path)
        module = importlib.import_module(f"{os.path.basename(path).rstrip('.py')}",
                                         package=f"{os.path.basename(import_path)}")
        sys.path.pop()
    return module


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
    meta: Any = None

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


# slurm shell commands
_SLURM_CMD_PUSH = ["sbatch"]
_SLURM_CMD_KILL = ["scancel"]
_SLURM_CMD_INFO = ["scontrol", "show", "job"]

# slurm script elements
_SLURM_SCRIPT_PREAMBLE = "#!/bin/bash"
_SLURM_SCRIPT_LINE_PREFIX = "#SBATCH"
_SLURM_SCRIPT_JOB_NAME = "--job-name"
_SLURM_SCRIPT_OUT_NAME = "--output"
_SLURM_SCRIPT_RESOURCES_MEM = "--mem"
_SLURM_SCRIPT_RESOURCES_TIME = "--time"
_SLURM_SCRIPT_RESOURCES_CPU = "--cpus-per-task"
_SLURM_SCRIPT_RESOURCES_GPU = "--gres"


def run_command(cmd: List[str]) -> str:
    """Execute a command in the shell.

    Args:
        cmd: list of str, the command with its arguments to execute.

    Returns:
        The standard output of the command.

    Raises:
        `OSError` if the standard error stream is not empty.
    """
    ps = subprocess.run(args=cmd, capture_output=True)
    if ps.stderr:
        raise OSError(f"Failed running '{cmd}' with error message: {ps.stderr}.")
    return ps.stdout.decode("utf-8")


class SlurmJobStatus(enum.Enum):
    """Some of the most frequently encountered slurm job statuses."""
    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
    UNKNOWN = 5


@dataclass(frozen=True)
class SlurmJob(Job):
    """A `Job` dataclass which redefines the behaviour when called.
    Instead of running a function or a `main` of a script, it runs an 'sbatch' command in the shell.

    Attributes:
        output_file: str, optional path to the file where the executed script will dump the result file.
    """
    output_file: str = None

    def __post_init__(self):
        if not isinstance(self.task, str):
            raise ValueError("Slurm job must be defined with a script to run.")
        super(SlurmJob, self).__post_init__()

    def __call__(self) -> 'Result':
        res = self._execute_job()
        return Result(id=self.id, data=res)

    def _execute_job(self) -> Any:
        with tempfile.NamedTemporaryFile(mode="w+t", suffix=".sh") as fp:
            contents = self._create_slurm_script()
            fp.writelines(contents)
            fp.seek(0)
            response = run_command(_SLURM_CMD_PUSH + [f"{fp.name}"])
        slurm_id = int(re.search(r"[\d]+", response).group())
        while True:
            slurm_status = self._query_job_status(slurm_id)
            if slurm_status in [SlurmJobStatus.RUNNING, SlurmJobStatus.PENDING]:
                time.sleep(1)
            elif slurm_status in [SlurmJobStatus.CANCELLED, SlurmJobStatus.FAILED]:
                return None
            elif slurm_status == SlurmJobStatus.COMPLETED:
                return self._fetch_result()
            else:
                raise RuntimeError(f"Unknown state of slurm job {slurm_id}.")

    def _create_slurm_script(self) -> List[str]:
        if not self.meta:
            raise ValueError(f"Cannot infer slurm job parameters. "
                             f"Fill in meta dict in job {self.id}.")
        else:
            # Preamble, job name and output log filename definitions
            content_lines = [
                f"{_SLURM_SCRIPT_PREAMBLE}\n",
                f"{_SLURM_SCRIPT_LINE_PREFIX} {_SLURM_SCRIPT_JOB_NAME}=job_{self.id}\n",
                f"{_SLURM_SCRIPT_LINE_PREFIX} {_SLURM_SCRIPT_OUT_NAME}=log_%j.txt\n"]
            # Resources specification
            n_cpus = int(self.meta.get("resources", {}).get("cpu", 1))
            if n_cpus >= 1:
                content_lines.append(f"{_SLURM_SCRIPT_LINE_PREFIX} {_SLURM_SCRIPT_RESOURCES_CPU}={n_cpus}\n")
            gpus = str(self.meta.get("resources", {}).get("gpu", ""))
            if gpus:
                if gpus.isnumeric():
                    gpus = f"gpu:{gpus}"
                content_lines.append(f"{_SLURM_SCRIPT_LINE_PREFIX} {_SLURM_SCRIPT_RESOURCES_GPU}={gpus}\n")
            mem = str(self.meta.get("resources", {}).get("memory", ""))
            if mem:
                content_lines.append(f"{_SLURM_SCRIPT_LINE_PREFIX} {_SLURM_SCRIPT_RESOURCES_MEM}={mem}\n")
            limit_time = str(self.meta.get("resources", {}).get("time", ""))
            if limit_time:
                content_lines.append(f"{_SLURM_SCRIPT_LINE_PREFIX} {_SLURM_SCRIPT_RESOURCES_TIME}={limit_time}\n")
            # Task specification
            binary = self.meta.get("binary", "python")
            content_lines.append(f"{binary} {self.task} {' '.join(map(str, self.args))} {self.output_file}")
        return content_lines

    @staticmethod
    def _query_job_status(slurm_id: int) -> SlurmJobStatus:
        response = run_command(_SLURM_CMD_INFO + [str(slurm_id)])
        job_state = re.search(r"JobState=(RUNNING|PENDING|COMPLETED|FAILED|CANCELLED)", response)
        if job_state is not None:
            job_state = job_state.group(1).lower()
            if job_state == "running":
                return SlurmJobStatus.RUNNING
            if job_state == "pending":
                return SlurmJobStatus.PENDING
            if job_state == "completed":
                return SlurmJobStatus.COMPLETED
            if job_state == "failed":
                return SlurmJobStatus.FAILED
            if job_state == "cancelled":
                return SlurmJobStatus.CANCELLED
            return SlurmJobStatus.UNKNOWN

    def _fetch_result(self) -> Any:
        if self.output_file is None:
            return None
        n_trials = 5
        for _ in range(n_trials):
            if os.path.isfile(self.output_file):
                break
            time.sleep(1)
        else:
            return None
        with open(self.output_file, 'rb') as fp:
            return pickle.load(fp)


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
