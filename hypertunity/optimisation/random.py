# -*- coding: utf-8 -*-

from typing import List, Tuple

import hypertunity.optimisation as ht_opt


class RandomSearch(ht_opt.BaseOptimiser):
    def __init__(self, domain, batch_size=1, seed=None):
        """Initialise the RandomSearch.

        If seed is provided the Domain is seeded.

        Args:
            domain: `Domain` of the objective to optimise. Will be sampled uniformly using the
                `sample()` method of the `Domain` object.
            batch_size: int, the number of samples to return at one step.
            seed: optional, int to seed the domain.
        """
        if seed is not None:
            domain = ht_opt.Domain(domain.as_dict(), seed=seed)
        super(RandomSearch, self).__init__(domain)
        self._batch_size = batch_size

    def run_step(self, *args, **kwargs) -> List[ht_opt.Sample]:
        """Sample uniformly the domain `self.batch_size` number of times."""
        return [self.domain.sample() for _ in range(self._batch_size)]

    def update(self, x, fx, **kwargs):
        """Update the RandomSearch optimiser's history track.

        This operation has no influence on the future samples, as they are all uniformly i.i.d. This implementation
        is only for completeness.

        Args:
            x: `Sample`, one sample of the domain of the objective function.
            fx: `EvaluationScore`, the evaluation score of the objective at `x`
        """
        if isinstance(x, ht_opt.Sample) and isinstance(fx, ht_opt.EvaluationScore):
            self.history.append(ht_opt.HistoryPoint(sample=x, metrics={"score": fx}))
        elif isinstance(x, (List, Tuple)) and isinstance(fx, (List, Tuple)) and len(x) == len(fx):
            self.history.extend([ht_opt.HistoryPoint(sample=i, metrics=j) for i, j in zip(x, fx)])
        else:
            raise ValueError("Update values for `x` and `f(x)` must be either "
                             "`Sample` and `EvaluationScore` or a list thereof.")
