"""Optimisation by a uniformly random search."""

from typing import List

from hypertunity.domain import Domain, Sample
from hypertunity.optimisation.base import Optimiser

__all__ = [
    "RandomSearch"
]


class RandomSearch(Optimiser):
    def __init__(self, domain: Domain, seed: int = None):
        """Initialise the RandomSearch.

        If seed is provided the Domain is seeded.

        Args:
            domain: `Domain` of the objective to optimise. Will be sampled uniformly using the
                `sample()` method of the `Domain` object.
            seed: optional, int to seed the domain.
        """
        if seed is not None:
            domain = Domain(domain.as_dict(), seed=seed)
        super(RandomSearch, self).__init__(domain)

    def run_step(self, batch_size=1, **kwargs) -> List[Sample]:
        """Sample uniformly the domain `batch_size` number of times.

        Args:
            batch_size: int, the number of samples to return at one step.

        Returns:
            A list of `batch_size` many samples.
        """
        return [self.domain.sample() for _ in range(batch_size)]
