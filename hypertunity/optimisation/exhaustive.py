# -*- coding: utf-8 -*-
"""Optimisation by exhaustive search, e.g. regular grid or list search.
"""

from .base import BaseOptimiser

from ..utils import support_american_spelling


class GridSearch(BaseOptimiser):

    @support_american_spelling
    def __init__(self, minimise):
        super(GridSearch, self).__init__(minimise=minimise)

    def run_step(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def reset(self):
        pass
