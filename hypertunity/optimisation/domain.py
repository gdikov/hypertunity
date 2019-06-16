# -*- coding: utf-8 -*-
"""Define the optimisation domain as a tweaked Python dictionary.
"""

import ast
import copy
import random
import os
import pickle

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
        self._ndim = 0
        for _, val in _deepiter_dict(self._data):
            if isinstance(val, list):
                self._is_continuous = True
            self._ndim += 1

    def __iter__(self):
        """For discrete domains iterate over the Cartesian product of all 1-dim subdomains.

        Raises:
            DomainNotIterableError in case of a continuous domain.

        Notes:
            All dimensions must be discrete (numeric or categorical).
            Even if one dimension is continuous the iteration will raise an error.
        """
        if self._is_continuous:
            raise DomainNotIterableError("The domain has a continuous subdomain and cannot be iterated.")

        def cartesian_walk(dct):
            if dct:
                key, vals = dct.popitem()
                if isinstance(vals, tuple):
                    for v in vals:
                        yield from (dict(**rem, **{key: v}) for rem in cartesian_walk(copy.deepcopy(dct)))
                elif isinstance(vals, dict):
                    for sub_v in cartesian_walk(copy.deepcopy(vals)):
                        yield from (dict(**rem, **{key: sub_v}) for rem in cartesian_walk(copy.deepcopy(dct)))
                else:
                    raise TypeError(f"Unexpected subdomain of type {type(vals)}.")
            else:
                yield {}

        yield from cartesian_walk(copy.deepcopy(self._data))

    def __eq__(self, other):
        """Compare all subdomains for equal bounds and sets. Order of subdomains is not important.
        """
        return self.as_dict() == other.as_dict()

    def __len__(self):
        """Compute the dimensionality of the domain.
        """
        return self._ndim

    def __getitem__(self, item):
        """Return the item (possibly subdomain) for a given key.
        """
        return self._data.__getitem__(item)

    def _validate(self):
        """Check for invalid domain specifications.
        """
        for keys, values in _deepiter_dict(self._data):
            if not (all(map(lambda x: isinstance(x, str), keys)) and isinstance(values, (tuple, list, dict))):
                return False
        return True

    def sample(self):
        """Draw a sample from the domain. All subdomains are sampled uniformly.
        """
        def sample_dict(dct):
            sample = {}
            for key, vals in dct.items():
                if isinstance(vals, tuple):
                    sample[key] = random.choice(vals)
                elif isinstance(vals, list):
                    sample[key] = random.uniform(*vals)
                else:
                    sample[key] = sample_dict(vals)
            return sample

        return sample_dict(self._data)

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
    """Iterate over all key, value pairs of a (possibly nested) dictionary. In this case, all keys of the
    nested dicts are summarised in a tuple.

    Args:
        dct: dict object to iterate.

    Yields:
        Tuple of keys (itself a tuple) and the corresponding value.

    Examples:
        >>> list(_deepiter_dict({"a": {"b": 1, "c": 2}, "d": 3}))
        >>> [(('a', 'b'), 1), (('a', 'c'), 2), (('d',), 3)]
    """
    def chained_keys_iter(prefix_keys, dct_tmp):
        for key, val in dct_tmp.items():
            chained_keys = prefix_keys + (key,)
            if isinstance(val, dict):
                yield from chained_keys_iter(chained_keys, val)
            else:
                yield chained_keys, val

    yield from chained_keys_iter((), dct)


class DomainNotIterableError(Exception):
    pass
