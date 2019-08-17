# -*- coding: utf-8 -*-
"""Experiment scheduler for running jobs locally in a parallel manner using joblib as a backend."""

from typing import List
import time
import multiprocessing as mp

import joblib

from .base import Scheduler
from .jobs import Job, Result
from hypertunity import utils

__all__ = [
    "LocalScheduler"
]


class LocalScheduler(Scheduler):
    """This Scheduler uses the locally available machine to run the jobs.
    A job can either be a python callable functions or a python executable script.
    """
    def __init__(self, n_parallel: int = None):
        """
        Args:
            n_parallel: int, the number of jobs that can be run in parallel.
        """
        super(LocalScheduler, self).__init__()
        if n_parallel is None:
            self.n_parallel = -2     # using all CPUs but 1
        else:
            self.n_parallel = max(n_parallel, 1)
        self._servant = mp.Process(target=self._run_servant)
        self._interrupt_event = mp.Event()
        self._servant.start()

    def __del__(self):
        """Clean up subprocess on object deletion."""
        super(LocalScheduler, self).__del__()
        if self._servant.is_alive():
            self._servant.terminate()

    def __enter__(self):
        """Run the servant subprocess on context creation."""
        super(LocalScheduler, self).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Interrupt and terminate the servant process."""
        super(LocalScheduler, self).__exit__(exc_type, exc_val, exc_tb)
        self._interrupt_event.set()
        # wait a bit for the subprocess to exit gracefully
        n_retries = 3
        while self._servant.is_alive() and n_retries > 0:
            n_retries -= 1
            time.sleep(0.05)
        self._servant.terminate()

    def _run_servant(self):
        """Run the pool of workers on the dispatched jobs, fetched from the job queue and
        collect the results into the result queue.

        Notes:
            The runner will take as long as all jobs from the job queue finish before any results are
            written to the result queue.
        """
        with joblib.Parallel(n_jobs=self.n_parallel) as parallel:
            while not self._interrupt_event.is_set():
                current_jobs = utils.drain_queue(self._job_queue)
                if not current_jobs:
                    continue
                # the order of the results corresponds to the that of the jobs
                # and the IDs don't need to be shuffled.
                ids = [job.id for job in current_jobs]
                # TODO: in a future version of joblib, this could be a generator and then the inputs
                #  would be stored immediately in the results queue. Be ready to update whenever
                #  this PR gets merged: https://github.com/joblib/joblib/pull/588
                results = parallel(joblib.delayed(job)() for job in current_jobs)
                assert len(ids) == len(results)
                for res in results:
                    self._result_queue.put_nowait(res)

    def dispatch(self, jobs: List[Job]):
        """Dispatch the jobs for parallel execution.

        Args:
            jobs: list of `Job`s to run.

        Notes:
            Although the jobs are scheduled to run immediately, the actual execution may take place later
            if the job runner is occupied with older jobs.
        """
        for job in jobs:
            self._job_queue.put_nowait(job)

    def collect(self, n_results: int = 0, timeout: float = None) -> List[Result]:
        """Collect the results from the finished jobs.

        Args:
            n_results: int, number of results to wait for. If n_results <= 0 then all available results
                will be returned.
            timeout: float, number of seconds (or fraction thereof) to wait if no results are available
                and `n_results` > 0.

        Returns:
            List of `Result` objects.
        """
        if n_results > 0:
            results = []
            for i in range(n_results):
                results.append(self._result_queue.get(block=True, timeout=timeout))
        else:
            results = utils.drain_queue(self._result_queue)
        return results

    def interrupt(self):
        """Interrupt the scheduler"""
        self._interrupt_event.set()
