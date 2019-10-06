"""A wrapper class for conducting multiple experiments, scheduling jobs and saving results."""

from typing import Type, Union, Callable

from hypertunity import optimisation
from hypertunity import reports
from hypertunity import utils
from hypertunity.domain import Domain
from hypertunity.optimisation import Optimiser
from hypertunity.reports import Reporter
from hypertunity.scheduling import Scheduler, Job, SlurmJob

__all__ = [
    "Trial"
]

OptimiserTypes = Union[str, Type[Optimiser], Optimiser]
ReporterTypes = Union[str, Type[Reporter], Reporter]


class Trial:
    """High-level API class for running hyperparameter optimisation.
    This class encapsulates optimiser querying, job building, scheduling and
    results collection as well as checkpointing and report generation.
    """

    @utils.support_american_spelling
    def __init__(self, objective: Union[Callable, str],
                 domain: Domain,
                 optimiser: OptimiserTypes = "bo",
                 reporter: ReporterTypes = "table",
                 device: str = "local",
                 **kwargs):
        """Initialise the `Trial` experiment manager.

        Args:
            objective: Callable or str, the objective function or script to run.
            domain: `Domain`, the domain of the objective function.
            optimiser: `Optimiser` or str, the optimiser method for domain sampling.
            reporter: `Reporter` or str, the reporting method for the results.
            device: str, the host device running the evaluations. Can be 'local' or 'slurm'.
            **kwargs: additional parameters for the optimiser, reporter and scheduler.
        """
        self.objective = objective
        self.domain = domain
        self.optimiser = self._init_optimiser(optimiser, **kwargs)
        self.reporter = self._init_reporter(reporter, **kwargs)
        self.scheduler = Scheduler
        self._timeout = kwargs.get("timeout", 259200.0)  # the number of seconds contained in 3 days
        self._job = self._init_job(device)

    def _init_optimiser(self, optimiser: OptimiserTypes, **kwargs) -> Optimiser:
        if isinstance(optimiser, str):
            optimiser_class = get_optimiser(optimiser)
        elif issubclass(optimiser, Optimiser):
            optimiser_class = optimiser
        elif isinstance(optimiser, Optimiser):
            return optimiser
        else:
            raise TypeError("An optimiser must be a either a string, "
                            "an Optimiser type or an Optimiser instance.")
        opt_kwargs = {}
        if "seed" in kwargs:
            opt_kwargs["seed"] = kwargs["seed"]
        return optimiser_class(self.domain, **opt_kwargs)

    def _init_reporter(self, reporter: ReporterTypes, **kwargs) -> Reporter:
        if isinstance(reporter, str):
            reporter_class = get_reporter(reporter)
        elif issubclass(reporter, Reporter):
            reporter_class = reporter
        elif isinstance(reporter, Reporter):
            return reporter
        else:
            raise TypeError("A reporter must be either a string, "
                            "a Reporter type or a Reporter instance.")
        rep_kwargs = {"metrics": kwargs.get("metrics", ["score"]),
                      "database_path": kwargs.get("database_path", ".")}
        if not issubclass(reporter_class, reports.Table):
            rep_kwargs["logdir"] = kwargs.get("logdir", "tensorboard/")
        return reporter_class(self.domain, **rep_kwargs)

    @staticmethod
    def _init_job(device: str) -> Type[Job]:
        device = device.lower()
        if device == "local":
            return Job
        if device == "slurm":
            return SlurmJob
        raise ValueError(f"Unknown device {device}. Select one from {{'local', 'slurm'}}.")

    def run(self, n_steps: int, n_parallel: int = 1, **kwargs):
        """Run the optimisation and objective function evaluation for a given number of steps.

        Args:
            n_steps: int, the total number of optimiser queries.
            n_parallel: int, the number of jobs that can be scheduled at once.
            **kwargs: additional keyword arguments for the optimisation, provided on the `run_step()` call.

        Keyword Args:
            batch_size: int, the number of samples that are proposed at once.
            minimise: bool, if the optimiser is `BayesianOptimisation` then this flag tells whether the objective
                function is being minimised or maximised.
        """
        batch_size = kwargs.get("batch_size", 1)
        n_parallel = min(n_parallel, batch_size)
        with self.scheduler(n_parallel=n_parallel) as scheduler:
            for i in range(n_steps):
                samples = self.optimiser.run_step(batch_size=batch_size, minimise=kwargs.get("minimise", False))
                jobs = [self._job(task=self.objective, args=s.as_dict()) for s in samples]
                scheduler.dispatch(jobs)
                evaluations = [r.data for r in scheduler.collect(n_results=batch_size, timeout=self._timeout)]
                self.optimiser.update(samples, evaluations)
                for s, e, j in zip(samples, evaluations, jobs):
                    self.reporter.log((s, e), meta={"job_id": j.id})


def get_optimiser(name: str) -> Type[Optimiser]:
    name = name.lower()
    if name.startswith(("bayes", "bo")):
        return optimisation.BayesianOptimisation
    if name.startswith("random"):
        return optimisation.RandomSearch
    if name.startswith(("grid", "exhaustive")):
        return optimisation.GridSearch
    raise ValueError(f"Unknown optimiser {name}. Select one from "
                     f"{{'bayesian_optimisation', 'random_search', 'grid_search'}}.")


def get_reporter(name: str) -> Type[Reporter]:
    name = name.lower()
    if name.startswith("table"):
        return reports.Table
    if name.startswith(("tensor", "tb")):
        import reports.tensorboard as tb
        return tb.Tensorboard
    raise ValueError(f"Unknown reporter {name}. Select one from {{'table', 'tensorboard'}}.")
