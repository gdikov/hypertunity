# -*- coding: utf-8 -*-
"""Common code and templates for the schedulers."""

import abc
import multiprocessing as mp

from typing import List

from hypertunity import utils
from .jobs import Job, Result

__all__ = [
    "Scheduler"
]


class Scheduler:
    """Base class for the schedulers. Maintains a `Job` and `Result` queues
    and defines template methods to override. This class should be used as
    a context manager.
    """
    def __init__(self):
        """Setup the job and results queues."""
        self._job_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._is_closed = False

    def __del__(self):
        """Close the queues and join all subprocesses before the object is deleted."""
        if not self._is_closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abc.abstractmethod
    def dispatch(self, jobs: List[Job]):
        """Schedule jobs for running. This method must be non-blocking.

        Args:
            jobs: list, `Job`s to schedule.
        """
        pass

    @abc.abstractmethod
    def collect(self, n_results: int, timeout: float = None) -> List[Result]:
        """Collect all the available results or wait until they become available.

        Args:
            n_results: int, number of results to collect.
            timeout: float, number of seconds to wait at most for a result to be collected.
                If None (default) then it will wait until all `n_results` are collected.

        Returns:
            A list of `Result` objects with length `n_results` at least.

        Notes:
            If `n_results` is overestimated and timeout is None, then this method will hang forever.
            Therefore it is recommended that a timeout is set.

        Raises:
            TimeoutError if more than `timeout` seconds elapse before a `Result` is collected.
        """
        pass

    @abc.abstractmethod
    def interrupt(self):
        """Interrupt the scheduler and all running jobs."""
        pass

    def close(self):
        """Close the queues and clean-up."""
        if not self._is_closed:
            utils.drain_queue(self._job_queue, close_queue=True)
            self._job_queue.join_thread()
            utils.drain_queue(self._result_queue, close_queue=True)
            self._result_queue.join_thread()
            self._is_closed = True
