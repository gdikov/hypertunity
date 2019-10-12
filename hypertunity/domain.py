"""Definition of the optimisation domain and a sample."""

import ast
import copy
import os
import pickle
import random
from collections import namedtuple
from typing import Tuple

__all__ = [
    "Domain",
    "DomainNotIterableError",
    "DomainSpecificationError",
    "Sample"
]


class _RecursiveDict:
    """Helper base class for the :class:`Domain` and :class:`Sample` classes.

    It implements common logic for creation, representation, type conversion and serialisation.
    """

    def __init__(self, dct):
        if isinstance(dct, dict):
            self._data = dct
        elif isinstance(dct, str):
            self._data = ast.literal_eval(dct)
        else:
            raise TypeError(
                f"A {self.__class__.__name__} object can be created from a Python dict or str objects only. "
                f"Unknown type {type(dct)} at initialisation.")

        self._ndim = 0
        for _, val in _deepiter_dict(self._data):
            self._ndim += 1

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        """Return the representation of the recursive dict using the string method."""
        return str(self)

    def __str__(self):
        """Return the string representation of the recursive dict."""
        return str(self._data)

    def __eq__(self, other):
        """Compare all subdomains for equal bounds and sets. The order of the subdomains is not important."""
        return self.as_dict() == other.as_dict()

    def __len__(self):
        """Compute the dimensionality of the recursive dict as the length of the flattened dict."""
        return self._ndim

    def __getitem__(self, item):
        """Return the item (possibly a subdomain) for a given key.

        Args:
            item: str of tuple of str. If the latter it will access nested structures with the next str in the tuple.
        """
        if isinstance(item, str):
            return self._data.__getitem__(item)
        elif isinstance(item, tuple) and all(map(lambda x: isinstance(x, str), item)):
            sub_dict = self._data
            for it in item:
                if not isinstance(sub_dict, dict):
                    raise KeyError(f"Unknown sub-key {it}.")
                sub_dict = sub_dict[it]
            return sub_dict

    def __add__(self, other: '_RecursiveDict'):
        """Merge self with the `other` :class:`_RecursiveDict`.

        Args:
            other: :class:`_RecursiveDict`. The recursive dictionary that will be merged into the current one.

        Returns:
            A new :class:`_RecursiveDict` object consisting of the subdomains of both domains.
            If the keys overlap and the subdomains are discrete or categorical, the values will be unified.

        Raises:
            :obj:`ValueError`: if identical keys point to different values.
        """
        flattened_a = self.flatten()
        flattened_b = other.flatten()
        # validate that the two _RecursiveDicts are disjoint
        if len(flattened_a.keys()) > len(flattened_a.keys() - flattened_b.keys()):
            raise ValueError(f"Ambiguous addition of {self.__class__.__name__} objects.")
        merged = list(flattened_a.items())
        merged.extend(list(flattened_b.items()))
        return self.__class__.from_list(merged)

    def flatten(self):
        """Return the flattened version of the recursive dict, i.e. without nested dicts.

        The keys of the nested subdomains are collected in a tuple to create a new unique key. For the sake of type
        consistency, the key of a non-nested subdomain is converted to a tuple with a single element.
        """
        return {keys: val for keys, val in _deepiter_dict(self._data)}

    def as_dict(self):
        """Convert the recursive dict object from :class:`_RecursiveDict` to :obj:`dict` type."""
        return copy.deepcopy(self._data)

    @classmethod
    def from_list(cls, lst):
        """Create a :class:`_RecursiveDict` object from a list of tuples.

        Args:
            lst: :obj:`List[Tuple]`. Each element is a pair of the keys (tuple of strings) and the value.

        Returns:
            A :class:`_RecursiveDict` object.

        Raises:
            :obj:`ValueError`: if the list contains duplicating keys with different values.

        Examples:
        ```python
            >>> lst = [(("a", "b"), {2, 3, 4}), (("c",), [0, 0.1])]
            >>> _RecursiveDict.from_list(lst)
            {"a": {"b": {2, 3, 4}}, "c": [0, 0.1]}
        ```
        """
        dct = {}
        head = dct
        for keys, vals in lst:
            if not keys:
                continue
            for k in keys[:-1]:
                if k not in dct:
                    dct[k] = {}
                dct = dct[k]
            if keys[-1] in dct and dct[keys[-1]] == vals:
                raise ValueError(f"Duplicating entries for keys {keys}.")
            dct[keys[-1]] = vals
            dct = head
        return cls(head)

    def serialise(self, filepath=None):
        """Serialise the :class:`_RecursiveDict` object to a file or a string if `filepath` is not supplied.

        Args:
            filepath: (optional) :obj:`str`. Filepath as to dump the serialised :class:`_RecursiveDict` object.

        Returns:
            The bytes representing the serialised :class:`_RecursiveDict` object.
        """
        serialised = pickle.dumps(self._data)
        if filepath is not None:
            with open(filepath, "wb") as fp:
                pickle.dump(self._data, fp)
        return serialised

    @classmethod
    def deserialise(cls, series):
        """Deserialise a serialised :class:`_RecursiveDict` object from a byte stream or file.

        Args:
            series: :obj:`str`. The serialised :class:`_RecursiveDict` object or a filepath to it.

        Returns:
            A :class:`_RecursiveDict` object.
        """
        if not isinstance(series, (bytes, bytearray)) and os.path.isfile(series):
            with open(series, "rb") as fp:
                return cls(pickle.load(fp))
        return cls(pickle.loads(series))

    def as_namedtuple(self):
        """Convert a :class:`_RecursiveDict` to a namedtuple type.

        Returns:
            A Python namedtuple object with names the same as the keys of the :class:`_RecursiveDict` dict.
            Nested dicts are accessed by successive attribute getters.

        Examples:
        ```python
            >>> rd = _RecursiveDict({"a": {"b": [1, 2]}, "c": {1, 2, 3}, "d": 2.})
            >>> nt = rd.as_namedtuple()
            >>> nt.a.b
            [1, 2]
            >>> nt.c == {1, 2, 3} and nt.d == 2.
            True
        ```
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


class Domain(_RecursiveDict):
    """Defines the optimisation domain of the objective function. It can be a continuous interval or a discrete
    set of numeric or non-numeric values. The latter is also designated as a categorical domain.
    It is represented as a Python dict object with the keys naming the variables and the values defining
    the set of allowed values.
    A :class:`Domain` can also be recursively specified. That is, a key can name a subdomain represented
    as a Python dict.

    For continuous sets use Python list to define an interval in the form [a, b], a < b.
    For discrete sets use Python sets, e.g. {1, 2, 5, -0.1} or {"option_a", "option_b"}.

    Examples:
        >>> simple_domain = {"x": {0, 1},
        >>>                  "y": [-1, 1],
        >>>                  "z": {-1, 2, 4}}
        >>> nested_domain = {"discrete": {"x": {1, 2, 3}, "y": {4, 5, 6}}
        >>>                  "continuous": {"x": [-4, 4], "y": [0, 1]}
        >>>                  "categorical": {"opt1", "opt2"}}
    """
    # Domain types
    Continuous = 1
    Discrete = 2
    Categorical = 3
    Invalid = 4

    def __init__(self, dct, seed=None):
        """Initialise the :class:`Domain`.

        Args:
            dct: :obj:`dict`. The mapping of variable names to sets of allowed values.
            seed: (optional) :obj:`int`. Seed for the randomised sampling.
        """
        super(Domain, self).__init__(dct)
        self._validate()
        self._rng = random.Random(seed)
        self._is_continuous = False
        for _, val in _deepiter_dict(self._data):
            if isinstance(val, list):
                self._is_continuous = True

    def __iter__(self):
        """Iterate over the domain if it is fully discrete.

        The iterations are over the Cartesian product of all 1-dim discrete subdomains.

        Raises:
            :class:`DomainNotIterableError`: if the domain has a at least one continuous subdomain.
        """
        if self._is_continuous:
            raise DomainNotIterableError("The domain has a continuous subdomain and cannot be iterated.")

        def cartesian_walk(dct):
            if dct:
                key, vals = dct.popitem()
                if isinstance(vals, set):
                    for v in vals:
                        yield from (dict(**rem, **{key: v}) for rem in cartesian_walk(copy.deepcopy(dct)))
                elif isinstance(vals, dict):
                    for sub_v in cartesian_walk(copy.deepcopy(vals)):
                        yield from (dict(**rem, **{key: sub_v}) for rem in cartesian_walk(copy.deepcopy(dct)))
                else:
                    raise TypeError(f"Unexpected subdomain of type {type(vals)}.")
            else:
                yield {}

        yield from map(Sample, cartesian_walk(copy.deepcopy(self._data)))

    def _validate(self):
        """Check for invalid domain specifications."""
        for keys, values in _deepiter_dict(self._data):
            if not (all(map(lambda x: isinstance(x, str), keys)) and isinstance(values, (set, list, dict))):
                raise DomainSpecificationError("Keys must be of type string and values "
                                               "must be either of type set, list or dict.")
            if isinstance(values, list) and (len(values) != 2 or values[0] >= values[1]):
                raise DomainSpecificationError("Interval must be specified by two numbers: [a, b], a < b.")

    def sample(self):
        """Draw a sample from the domain. All subdomains are sampled uniformly.

        Returns:
            A :class:`Sample` object.
        """

        def sample_dict(dct):
            sample = {}
            for key, vals in dct.items():
                if isinstance(vals, set):
                    sample[key] = self._rng.choice(list(vals))
                elif isinstance(vals, list):
                    sample[key] = self._rng.uniform(*vals)
                else:
                    sample[key] = sample_dict(vals)
            return sample

        return Sample(sample_dict(self._data))

    @property
    def is_continuous(self):
        """Return `True` if at least one subdomain is continuous."""
        return self._is_continuous

    @classmethod
    def get_type(cls, subdomain):
        """Return the type of the set of values in a subdomain.

        Args:
            subdomain: one of :obj:`dict`, :obj:`list` or :obj:`set`. The subdomain to get the type for.

        Returns:
            One of `Domain.Continuous`, `Domain.Discrete`, `Domain.Categorical` or `Domain.Invalid`.
        """

        def is_numeric(x):
            try:
                float(x)
            except ValueError:
                return False
            return True

        if isinstance(subdomain, list):
            return Domain.Continuous
        if isinstance(subdomain, set):
            if all(map(is_numeric, subdomain)):
                return Domain.Discrete
            return Domain.Categorical
        return Domain.Invalid

    def split_by_type(self) -> Tuple['Domain', 'Domain', 'Domain']:
        """Split the domain into discrete, categorical and continuous subdomains respectively.

        Returns:
            A tuple of three :class:`Domain` objects for the discrete numerical, categorical
            and continuous subdomains.
        """
        discrete, categorical, continuous = [], [], []
        for keys, vals in self.flatten().items():
            if Domain.get_type(vals) == Domain.Continuous:
                continuous.append((keys, vals))
            elif Domain.get_type(vals) == Domain.Categorical:
                categorical.append((keys, vals))
            elif Domain.get_type(vals) == Domain.Discrete:
                discrete.append((keys, vals))
            else:
                raise ValueError("Encountered an invalid subdomain.")
        return Domain.from_list(discrete), Domain.from_list(categorical), Domain.from_list(continuous)


class DomainNotIterableError(TypeError):
    """Alias for the :obj:`TypeError` raised during iteration of (partially) continuous :class:`Domain` object."""
    pass


class DomainSpecificationError(ValueError):
    """Alias for the :obj:`ValueError` raised during :class:`Domain` object creation from an invalid set of values."""
    pass


class Sample(_RecursiveDict):
    """Defines a sample from the optimisation domain.

    It has the same recursive structure a :class:`Domain` object, however each dimension is represented by one
    value only. The keys are exactly as the keys of the respective domain.

    Examples:
        >>> domain = Domain({"x": {"y": {0, 1, 2}}, "z": [3, 4]})
        >>> domain.sample()
        {'x': {'y': 0}, 'z': 3.1415926535897932}
    """

    def __init__(self, dct):
        """Initialise the :class:`Sample` object from a dict."""
        super(Sample, self).__init__(dct)

    def __iter__(self):
        """Iterate over all values in the sample.

        Yields:
            A tuple of keys and a single value, where the keys are a tuple of strings.
        """
        yield from self.flatten().items()


def _deepiter_dict(dct):
    """Iterate over all key, value pairs of a (possibly nested) dictionary. In this case, all keys of the
    nested dicts are summarised in a tuple.

    Args:
        dct: dict object to iterate.

    Yields:
        Tuple of keys (itself a tuple) and the corresponding value.

    Examples:
        >>> list(_deepiter_dict({"a": {"b": 1, "c": 2}, "d": 3}))
        [(('a', 'b'), 1), (('a', 'c'), 2), (('d',), 3)]
    """

    def chained_keys_iter(prefix_keys, dct_tmp):
        for key, val in dct_tmp.items():
            chained_keys = prefix_keys + (key,)
            if isinstance(val, dict):
                yield from chained_keys_iter(chained_keys, val)
            else:
                yield chained_keys, val

    yield from chained_keys_iter((), dct)
