# -*- coding: utf-8 -*-

import pytest

import numpy as np

from hypertunity.optimisation.domain import Domain
from hypertunity.optimisation.base import EvaluationScore
from .. import gpyopt


def test_gpyopt_bo_update():
    domain = Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    bo = gpyopt.GPyOptBackend(domain, minimise=True)
    sample = bo.run_step()
    n_reps = 3
    for i in range(n_reps):
        bo.update(sample, EvaluationScore(2. * i))
    assert len(bo._data_x) == n_reps and len(bo._data_fx) == n_reps
    assert np.all(bo._data_x == np.tile(bo._convert_to_gpyopt_sample(sample), (n_reps, 1)))
    assert np.all(bo._data_fx == 2. * np.arange(n_reps).reshape(n_reps, 1))


def test_gpyopt_bo_reset():
    domain = Domain({"a": {"b": [2, 3]}, "c": [0, 0.1]})
    bo = gpyopt.GPyOptBackend(domain, minimise=True)
    sample = bo.run_step()
    n_reps = 3
    for i in range(n_reps):
        bo.update(sample, EvaluationScore(2. * i))
    bo.reset()
    assert len(bo.history) == 0


def test_clean_and_join_and_revert():
    strings = [("vxc", "", "", "___"), ("_", "_", ""), ("asd",), ("asd", "dxcv")]
    for s in strings:
        assert s == gpyopt._revert_clean_and_join(gpyopt._clean_and_join(s))
    with pytest.raises(ValueError):
        gpyopt._clean_and_join(["asd", "\\", "\n"])
