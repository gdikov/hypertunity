"""Optimisation by exhaustive search, aka grid search."""

from typing import List

from hypertunity.domain import Domain, Sample, DomainNotIterableError
from hypertunity.optimisation.base import Optimiser, ExhaustedSearchSpaceError

__all__ = [
    "GridSearch"
]


class GridSearch(Optimiser):
    """Grid search pseudo-optimiser."""

    def __init__(self, domain: Domain, sample_continuous: bool = False, seed: int = None):
        """Initialise the :class:`GridSearch` optimiser from a discrete domain.

        If the domain contains continuous subspaces, then they could be sampled if `sample_continuous` is enabled.

        Args:
            domain: :class:`Domain`. The domain to iterate over.
            sample_continuous: (optional) :obj:`bool`. Whether to sample the continuous subspaces of the domain.
            seed: (optional) :obj:`int`. Seed for the sampling of the continuous subspace if necessary.
        """
        if domain.is_continuous and not sample_continuous:
            raise DomainNotIterableError(
                "Cannot perform grid search on (partially) continuous domain. "
                "To enable grid search in this case, set 'sample_continuous' to True.")
        super(GridSearch, self).__init__(domain)
        discrete_domain, categorical_domain, continuous_domain = domain.split_by_type()
        # unify the discrete and the categorical into one, as they can be iterated:
        self.discrete_domain = discrete_domain + categorical_domain
        if seed is not None:
            self.continuous_domain = Domain(continuous_domain.as_dict(), seed=seed)
        else:
            self.continuous_domain = continuous_domain
        self._discrete_domain_iter = iter(self.discrete_domain)
        self._is_exhausted = len(self.discrete_domain) == 0
        self.__exhausted_err = ExhaustedSearchSpaceError(
            "The domain has been exhausted. Reset the optimiser to start again.")

    def run_step(self, batch_size: int = 1, **kwargs) -> List[Sample]:
        """Get the next `batch_size` samples from the Cartesian-product walk over the domain.

        Args:
            batch_size: (optional) :obj:`int`. The number of samples to suggest at once.

        Returns:
            A list of :class:`Sample` instances from the domain.

        Raises:
            :class:`ExhaustedSearchSpaceError`: if the (discrete part of the) domain is fully exhausted and
                no samples can be generated.

        Notes:
            This method does not guarantee that the returned list of :class:`Samples` will be of length `batch_size`.
            This is due to the size of the domain and the fact that samples will not be repeated.
        """
        if self._is_exhausted:
            raise self.__exhausted_err

        samples = []
        for i in range(batch_size):
            try:
                discrete = next(self._discrete_domain_iter)
            except StopIteration:
                self._is_exhausted = True
                break
            if self.continuous_domain:
                continuous = self.continuous_domain.sample()
                samples.append(discrete + continuous)
            else:
                samples.append(discrete)
        if samples:
            return samples
        raise self.__exhausted_err

    def reset(self):
        """Reset the optimiser to the beginning of the Cartesian-product walk."""
        super(GridSearch, self).reset()
        self._discrete_domain_iter = iter(self.discrete_domain)
        self._is_exhausted = len(self.discrete_domain) == 0
