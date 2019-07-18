# -*- coding: utf-8 -*-
"""Optimisation by exhaustive search, e.g. regular grid or list search."""

from typing import List

import hypertunity.optimisation as ht_opt


__all__ = [
    "GridSearch",
    "ListSearch",
    "ExhaustedSearchSpaceError"
]


class ExhaustedSearchSpaceError(Exception):
    pass


class GridSearch(ht_opt.BaseOptimiser):
    def __init__(self, domain, batch_size=1, sample_continuous=False, seed=None):
        """Initialise the GridSearch optimiser from a discrete domain.

        If the domain contains continuous subspaces, then they could be sampled if `sample_continuous` is enabled.

        Args:
            domain: `Domain`, the domain to iterate over.
            batch_size: int, the number of samples to supply at each step.
            sample_continuous: bool, whether to sample the continuous subspaces of the domain.
            seed: optional int, seed the sampling of the continuous subspace.
        """
        if domain.is_continuous and not sample_continuous:
            raise ht_opt.DomainNotIterableError(
                "Cannot perform grid search on (partially) continuous domain. "
                "To enable grid search in this case, set 'sample_continuous' to True.")
        super(GridSearch, self).__init__(domain)
        discrete_domain, categorical_domain, continuous_domain = ht_opt.split_domain_by_type(domain)
        # unify the discrete and the categorical into one, as they can be iterated:
        self.discrete_domain = discrete_domain + categorical_domain
        if seed is not None:
            self.continuous_domain = ht_opt.Domain(continuous_domain.as_dict(), seed=seed)
        else:
            self.continuous_domain = continuous_domain
        self._discrete_domain_iter = iter(self.discrete_domain)
        self._is_exhausted = len(self.discrete_domain) == 0
        self._batch_size = batch_size
        self.__exhausted_err = ExhaustedSearchSpaceError(
            "The domain has been exhausted. Reset the optimiser to start again.")

    def run_step(self) -> List[ht_opt.Sample]:
        """Get the next `batch_size` samples from the Cartesian-product walk over the domain.

        Returns:
            A list of `Sample`s from the domain.

        Raises:
            `ExhaustedSearchSpaceError` if the (discrete part of the) domain is fully exhausted and
            no samples can be generated.

        Notes:
            This method does not guarantee that the returned list of Samples will be of a `batch_size` size.
            This is due to the fixed size of the domain. If it gets exhausted during batch generation,
            the method will return the remaining samples to be evaluated.
        """
        if self._is_exhausted:
            raise self.__exhausted_err

        samples = []
        for i in range(self._batch_size):
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


class ListSearch:
    pass
