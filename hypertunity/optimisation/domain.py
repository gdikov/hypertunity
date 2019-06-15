# -*- coding: utf-8 -*-
"""Define the optimisation domain and a sample of it.
"""

import ast
import pickle
import os

from collections import namedtuple


class Domain:
    """Define the optimisation domain which can be of continuous or discrete numeric nature or a set of
    other categorical non-numeric values. It is represented as a Python dict object with the keys naming the
    subdomains and the values defining the set of possible variable values. For continuous sets use lists: [min, max].
    For discrete (categorical or numeric) sets use tuples, e.g. (1, 2, 5, -0.1) or ("1", "2").

    Notes:
        Domains can be recursively specified, i.e. a key can be a name of a subdomain and the value---a Python dict.

    Examples:
        >>> three_dim_domain = {"x": (0, 1),
        >>>                     "y": [-2.5, 3.5],
        >>>                     "z": [-1, 2, 4]}
        >>> nested_domain = {"discrete_subdomain": {"x": (1, 2, 3), "y": (4, 5, 6)}
        >>>                  "continuous_subdomain": {"x": [-4, 4], "y": [0, 1]}
        >>>                  "categorical_options": ("opt1", "opt2")}
    """
    def __init__(self, dct):
        """Initialise the domain object and compute cached properties.

        Args:
            dct: dict object specifying the domain.
        """
        if isinstance(dct, dict):
            self._data = dct
        elif isinstance(dct, str):
            self._data = ast.literal_eval(dct)
        else:
            raise TypeError(f"A 'Domain' object can be created from a Python dict or str objects only. "
                            f"Unknown type {type(dct)} for initialisation.")

        if not self._validate():
            raise ValueError("Bad domain specification. Keys must be of type string and values must be "
                             "either of a tuple, list or dict type.")

        self._is_continuous = False
        for _, val in _deepiter_dict(self._data):
            if isinstance(val, list):
                self._is_continuous = True
                break

    def __iter__(self):
        raise NotImplementedError

    def __eq__(self, other):
        return self.as_dict() == other.as_dict()

    def _validate(self):
        """Check for invalid domain specifications.
        """
        for keys, values in _deepiter_dict(self._data):
            if not (all(map(lambda x: isinstance(x, str), keys)) and isinstance(values, (tuple, list, dict))):
                return False
        return True

    @property
    def is_continuous(self):
        """Return `True` if at least one subdomain is continuous.
        """
        return self._is_continuous

    def serialise(self, filepath=None):
        """Serialise the `Domain` object to a file or a string (if `filepath` is not supplied)

        Args:
            filepath: optional filepath as str to dump the serialised `Domain` object.

        Returns:
            The bytes representing the serialised `Domain` object.
        """
        serialised = pickle.dumps(self._data)
        if filepath is not None:
            with open(filepath, "wb") as fp:
                pickle.dump(self._data, fp)
        return serialised

    @classmethod
    def deserialise(cls, series):
        """Deserialise a serialised `Domain` object from a byte stream or file.

        Args:
            series: str representing the serialised `Domain` object or a filepath.

        Returns:
            A corresponding `Domain` object.
        """
        if not isinstance(series, (bytes, bytearray)) and os.path.isfile(series):
            with open(series, "rb") as fp:
                return cls(pickle.load(fp))
        return cls(pickle.loads(series))

    def as_dict(self):
        """Convert the domain object from a `Domain` to dict type."""
        return self._data

    def as_namedtuple(self):
        """Convert the domain object from a `Domain` to a namedtuple type.

        Returns:
            A Python namedtuple object with names the same as the keys of the `Domain` dict. Nested domains
            are accessed by successive attribute getters.

        Examples:
            >>> dom = Domain({"a": {"b": [1, 2]}, "c": (1, 2, 3)})
            >>> nt = dom.as_namedtuple()
            >>> nt.a.b
            >>> [1, 2]
            >>> nt.c == (1, 2, 3)
            >>> True
        """
        def helper(dct):
            keys, vals = [], []
            for k, v in dct.items():
                keys.append(k)
                if isinstance(v, dict):
                    vals.append(helper(v))
                else:
                    vals.append(v)
            # The dict.keys() and dict.values() will iterate in the same order as long as dct is not modified.
            return namedtuple("NT_" + self.__class__.__name__, keys)(*vals)
        return helper(self._data)


def _deepiter_dict(dct):
    def chained_keys_iter(prefix_keys, dct_tmp):
        for key, val in dct_tmp.items():
            chained_keys = prefix_keys + (key,)
            if isinstance(val, dict):
                yield from chained_keys_iter(chained_keys, val)
            else:
                yield chained_keys, val

    yield from chained_keys_iter((), dct)
