"""Optimisation by a uniformly random search."""

from typing import List

from hypertunity.domain import Domain, Sample
from hypertunity.optimisation.base import Optimiser

__all__ = [
    "RandomSearch"
]


class RandomSearch(Optimiser):
    """Uniform random sampling pseudo-optimiser."""

    def __init__(self, domain: Domain, seed: int = None):
        """Initialise the :class:`RandomSearch` search space.

        Args:
            domain: :class:`Domain`. The domain of the objective function.
                It will be sampled uniformly using the :py:meth:`sample()`
                method of the :class:`Domain`.
            seed: (optional) :obj:`int`. The seed for the domain sampling.
        """
        if seed is not None:
            domain = Domain(domain.as_dict(), seed=seed)
        super(RandomSearch, self).__init__(domain)

    def run_step(self, batch_size=1, **kwargs) -> List[Sample]:
        """Sample uniformly the domain for `batch_size` number of times.

        Args:
            batch_size: (optional) :obj:`int`. The number of samples to return
                at one step.

        Returns:
            A list of `batch_size` many :class:`Sample` instances.
        """
        return [self.domain.sample() for _ in range(batch_size)]
