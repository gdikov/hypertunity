"""Definition of `Job` and `Result` classes used to encapsulate an experiment
and the corresponding outcomes.
"""

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
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

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
        This function should be used with care as it will allow for jobs with
        repeating IDs to be created. As a consequence, two or more
        :class:`Result` objects might coexist end make the actual experiment
        outcome ambiguous.
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
        path: :obj:`str`, can be either a module import of the form
            [package.]*[module] if the outer most package is in the
            `PYTHONPATH`, or a path to an arbitrary python script.

    Returns:
        The loaded python script as a module.
    """
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot find script {path}.")
        if not os.path.basename(path).endswith(".py"):
            raise ValueError(

                f"Expected a python script ending with *.py, "
                f"found {os.path.basename(path)}.")
        import_path = os.path.dirname(os.path.abspath(path))
        sys.path.append(import_path)
        module = importlib.import_module(
            f"{os.path.basename(path).rstrip('.py')}",
            package=f"{os.path.basename(import_path)}"
        )
        sys.path.pop()
    return module


def run_command(cmd: List[str]) -> str:
    """Execute a command in the shell.

    Args:
        cmd: :obj:`List[str]`. The command with its arguments to execute.

    Returns:
        The standard output of the command.

    Raises:
        :obj:`OSError`: if the standard error stream is not empty.
    """
    ps = subprocess.run(args=cmd, capture_output=True)
    if ps.stderr:
        raise OSError(f"Failed running {' '.join(cmd)} with error message: "
                      f"{ps.stderr.decode('utf-8')}.")
    return ps.stdout.decode("utf-8")


def get_callable_from_script(script_path: str, func_name: str = "main") -> Callable:
    """Convert a module to a callable function and call the `main` function of
    the module.

    Args:
        script_path: str, the file path to the python script to run. It can
            either be given as a module i.e. in the [package.]*[module] form,
            or as a path to a *.py file in case it is not added into the
            PYTHONPATH environment variable.
        func_name: str, the name of the function to run.

    Returns:
        The wrapper which calls a function from the script module.

    Raises:
          `AttributeError` if the script does not define a `func_name` function.
    """

    def wrapper(*args):
        module = import_script(script_path)
        if not hasattr(module, func_name):
            raise AttributeError(
                f"Cannot find {func_name} function in {script_path}."
            )
        return getattr(module, func_name)(*args)

    return wrapper


def run_script_with_args(binary: str, script_path: str, *args: Any, **kwargs: Any):
    """Run script using a binary and command line arguments.

    Args:
        binary: str, the binary to run the script with, e.g. 'python'.
        script_path: str, the path to the script.
        *args: Any, a collection of arguments which will be converted to string
            and passed on to the run command.
        **kwargs: Any, keyword arguments which will be converted to named script
            arguments.

    Returns:
        The contents of the results, which the script is assumed to store,
        given an output file path as an argument.

    Raises:
        FileNotFoundError if the script cannot be found.

    Notes:
        It assumes that the script will store the results on disk using the
        path provided by the last of the command line arguments.
    """
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Cannot find script {script_path}.")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "results.pkl")
        args_as_str, kwargs_as_str = [], []
        if args:
            args_as_str.extend([*map(str, args), output_file])
        if kwargs:
            kwargs_as_str.extend([
                str(item) for k_v in kwargs.items() for item in k_v
            ])
            kwargs_as_str.extend(["--output_file", output_file])
        run_command([binary, script_path, *args_as_str, *kwargs_as_str])
        return fetch_result(output_file)


def fetch_result(output_file, n_trials: int = 5, waiting_time: float = 1.0) -> Any:
    """Load the output file.

    Args:
        output_file: str, a path to the output file.
        n_trials: int, optional number of trials to load the file, afterwards a
            None is returned.
        waiting_time: float, time in seconds to wait before retrying to load
            the file.

    Returns:
        The unpickled output file if found, else None.
    """
    if output_file is None:
        return None
    for _ in range(n_trials):
        if os.path.isfile(output_file):
            break
        time.sleep(waiting_time)
    else:
        return None
    with open(output_file, 'rb') as fp:
        return pickle.load(fp)


@dataclass(frozen=True)
class Job:
    """Default :class:`Job` class defining an experiment as a runnable task on
    the local machine.

    The job is defined by a callable function or a script task. In the case of
    the former the `args` will be passed directly to it upon calling. Otherwise
    either a module will be run as a scirpt with command line arguments or a
    function, attribute of the module, will be called with the `args` as input.
    In both cases a :class:`Result` object will be returned.

    Attributes:
        id: :obj:`int`. The job identifier. Must be unique.
        args: :obj:`tuple` or :obj:`dict`. The arguments or keyword arguments
            for the callable function or script.
        task: :obj:`Callable` or :obj:`str`, a python function to run or a
            file path to a python script.
    """
    task: Union[Callable, str]
    args: Union[Tuple, Dict] = ()
    id: int = field(default_factory=generate_id)
    meta: Any = None

    # job related constants
    _JOB_SCRIPT_FUNC_SEPARATOR = ":"
    _JOB_DEFAULT_BINARY = "source"
    _JOB_SCRIPT_FUNC_SEPARATION_REGEX = r"[^\w\/\.]+"

    def __post_init__(self):
        if not isinstance(self.task, (Callable, str)):
            raise ValueError(
                "Job's task must be either a callable function "
                "or a path to a script."
            )
        if self.id in _JOB_REGISTRY:
            raise ValueError(
                f"Job with an ID {self.id} is already created. "
                f"Reusing IDs is prohibited."
            )
        _JOB_REGISTRY.add(self.id)

    def __hash__(self):
        return hash(str(self.id))

    def __call__(self, *args, **kwargs) -> 'Result':
        all_args = args
        all_kwargs = kwargs
        if isinstance(self.args, Tuple):
            all_args += self.args
        else:
            all_kwargs = dict(**kwargs, **self.args)
        if isinstance(self.task, Callable):
            runnable = self.task
        else:
            runnable = self._build_callable()
        return Result(id=self.id, data=runnable(*all_args, **all_kwargs))

    def _build_callable(self):
        """Create a function from a string task.

        If the task is of the form /path/to/script.py::func_to_run, split the
        path from the func and return a script.func_to_run callable.
        If the task is of the form /path/to/script.py, then return a
        python /path/to/script.py callable.
        """
        if self._JOB_SCRIPT_FUNC_SEPARATOR in self.task:
            # split the task string by the [:]+ marker
            script_path, func_name = re.split(
                self._JOB_SCRIPT_FUNC_SEPARATION_REGEX, self.task
            )
            assert script_path and func_name, \
                f"Empty path {script_path} or function name {func_name}"
            runnable = get_callable_from_script(script_path, func_name)
        else:
            binary = self._infer_binary()
            runnable = partial(run_script_with_args, binary, self.task)
        return runnable

    def _infer_binary(self):
        if isinstance(self.meta, dict) and "binary" in self.meta:
            return self.meta["binary"]
        if self.task.endswith(".py"):
            return "python"
        if self.task.endswith(".sh"):
            return "bash"
        return self._JOB_DEFAULT_BINARY


class SlurmJobState(enum.Enum):
    """Some of the most frequently encountered slurm job statuses."""

    PENDING = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4
    UNKNOWN = 5

    @classmethod
    def from_string(cls, state: str):
        if state == "running":
            return cls.RUNNING
        if state == "pending":
            return cls.PENDING
        if state == "completed":
            return cls.COMPLETED
        if state == "failed":
            return cls.FAILED
        if state == "cancelled":
            return cls.CANCELLED
        return cls.UNKNOWN


@dataclass(frozen=True)
class SlurmJob(Job):
    """A :class:`Job` subclass to schedule tasks on Slurm.

    Runs an 'sbatch' command in the shell with the script.

    Attributes:
        output_file: (optional) :obj:`str`. Path to the file where the executed
            script will dump the result file. If none is provided, a temporary
            file will be created.
    """

    output_file: str = None

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

    # other macros
    _SLURM_JOB_STATE_REGEX = r"JobState=(RUNNING|PENDING|COMPLETED|FAILED|CANCELLED)"

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
            response = run_command(self._SLURM_CMD_PUSH + [f"{fp.name}"])
        slurm_id = int(re.search(r"[\d]+", response).group())
        while True:
            slurm_status = self._query_job_status(slurm_id)
            if slurm_status in [SlurmJobState.RUNNING, SlurmJobState.PENDING]:
                time.sleep(1)
            elif slurm_status in [SlurmJobState.CANCELLED, SlurmJobState.FAILED]:
                return None
            elif slurm_status == SlurmJobState.COMPLETED:
                return fetch_result(self.output_file)
            else:
                raise RuntimeError(f"Unknown state of slurm job {slurm_id}.")

    def _create_slurm_script(self) -> List[str]:
        if not self.meta:
            raise ValueError(f"Cannot infer slurm job parameters. "
                             f"Fill in meta dict in job {self.id}.")
        else:
            # Preamble, job name and output log filename definitions
            content_lines = [
                f"{self._SLURM_SCRIPT_PREAMBLE}\n",
                f"{self._SLURM_SCRIPT_LINE_PREFIX} "
                f"{self._SLURM_SCRIPT_JOB_NAME}=job_{self.id}\n",
                f"{self._SLURM_SCRIPT_LINE_PREFIX} "
                f"{self._SLURM_SCRIPT_OUT_NAME}=log_%j.txt\n"]
            # Resources specification
            n_cpus = int(self.meta.get("resources", {}).get("cpu", 1))
            if n_cpus >= 1:
                content_lines.append(
                    f"{self._SLURM_SCRIPT_LINE_PREFIX} "
                    f"{self._SLURM_SCRIPT_RESOURCES_CPU}={n_cpus}\n"
                )
            gpus = str(self.meta.get("resources", {}).get("gpu", ""))
            if gpus:
                if gpus.isnumeric():
                    gpus = f"gpu:{gpus}"
                content_lines.append(
                    f"{self._SLURM_SCRIPT_LINE_PREFIX} "
                    f"{self._SLURM_SCRIPT_RESOURCES_GPU}={gpus}\n"
                )
            mem = str(self.meta.get("resources", {}).get("memory", ""))
            if mem:
                content_lines.append(
                    f"{self._SLURM_SCRIPT_LINE_PREFIX} "
                    f"{self._SLURM_SCRIPT_RESOURCES_MEM}={mem}\n"
                )
            limit_time = str(self.meta.get("resources", {}).get("time", ""))
            if limit_time:
                content_lines.append(
                    f"{self._SLURM_SCRIPT_LINE_PREFIX} "
                    f"{self._SLURM_SCRIPT_RESOURCES_TIME}={limit_time}\n"
                )
            # Task specification
            binary = self.meta.get("binary", "python")
            if isinstance(self.args, Tuple):
                # build positional arguments
                script_args = ' '.join([*map(str, self.args), self.output_file])
            else:
                # build named arguments
                script_args = ' '.join([
                    *(str(item)
                      for key_val in self.args.items()
                      for item in key_val),
                    "--output_file", self.output_file
                ])
            content_lines.append(f"{binary} {self.task} {script_args}")
        return content_lines

    def _query_job_status(self, slurm_id: int) -> SlurmJobState:
        response = run_command(self._SLURM_CMD_INFO + [str(slurm_id)])
        job_state = re.search(self._SLURM_JOB_STATE_REGEX, response)
        if job_state is not None:
            job_state = job_state.group(1).lower()
            return SlurmJobState.from_string(job_state)


@dataclass(frozen=True)
class Result:
    """A :class:`Result` class to store the output of the executed :class:`Job`.

     It shares the same id as the job which generated it.

    Attributes:
        id: :obj:`int`. The identifier of the `Result` object which corresponds
            to the job that has been run.
        data: :obj:`Any`. The output data of the job.
    """
    data: Any
    id: int

    def __post_init__(self):
        if self.id in _RESULT_REGISTRY:
            raise ValueError(
                f"Result with an ID {self.id} is already created. "
                f"Reusing IDs is prohibited."
            )
        _RESULT_REGISTRY.add(self.id)
