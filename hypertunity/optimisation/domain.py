# -*- coding: utf-8 -*-
"""Define the optimisation domain and a sample of it.
"""

import os
import json
import copy

from collections import namedtuple


class Domain:
    """Define the optimisation domain which can be of continuous or discrete numeric nature or a set of
    other categorical non-numeric values. It is represented as a JSON object with the keys naming the subdomains
    and the values defining the set of possible values. For continuous sets use tuples: (min, max).
    For discrete (categorical or numeric) sets use lists, e.g. [1, 2, 5, -0.1] or ["1", "2"].

    Notes:
        Domains can be recursively specified, i.e. a key can be a name of a subdomain.

    Examples:
        three_dim_domain = {"x": (0, 1),
                            "y": (-2.5, 3.5),
                            "z": [-1, 2, 4]
                           }

        nested_domain = {"discrete_subdomain": {"x": [1, 2, 3], "y": [4, 5, 6]}
                         "continuous_subdomain": {"x": (-4, 4), "y": (0, 1)}
                         "categorical_options": ["opt1", "opt2"]}
    """
    def __init__(self, json_obj):
        self._data = json_obj
        self._is_continuous = False
        for _, val in _deepiter_dict(self._data):
            if isinstance(val, tuple):
                self._is_continuous = True

    def __iter__(self):
        pass

    @property
    def is_continuous(self):
        """Return `True` if at least one subdomain is continuous.
        """
        return self._is_continuous

    def serialise(self, filepath=None):
        """Serialise the `Domain` object to a file or a string (if `filepath` is not supplied)

        Args:
            filepath: optional filepath as str to dump the serialised JSON domain.

        Returns:
            The string representing the serialised JSON domain.
        """
        serialised = json.dumps(self._data)
        if filepath is not None:
            with open(filepath, "w") as fp:
                json.dump(serialised, fp)
        return serialised

    @classmethod
    def deserialise(cls, series):
        """Deserialise a serialised JSON domain object.

        Args:
            series: str representing the serialised JSON or a filepath.

        Returns:
            A corresponding `Domain` object.
        """
        if os.path.isfile(series):
            with open(series, "r") as fp:
                series = fp
        return cls(json.load(series))

    def as_namedtuple(self):
        """Convert the domain from a JSON dict type to a namedtuple type.

        Returns:
            A Python object with attributes the keys of the serialised JSON object.
        """
        cls_name = self.__class__.__name__
        serialised = self.serialise()
        # Note that the x.keys() and x.values() will iterate in the same order as long as x is not modified.
        return json.loads(serialised, object_hook=lambda x: namedtuple(cls_name, x.keys(), rename=True)(*x.values()))


def _deepiter_dict(dict_obj):
    def chained_keys_iter(prefix_keys, dict_):
        for key, val in dict_.items():
            chained_keys = prefix_keys + (key,)
            if isinstance(val, dict):
                yield from chained_keys_iter(chained_keys, val)
            else:
                yield chained_keys, val

    yield from chained_keys_iter((), dict_obj)
