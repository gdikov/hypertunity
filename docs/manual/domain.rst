Domain
======

The set of all hyperparameters and the corresponding ranges of possible values is specified using the :class:`Domain` class.
It can be initialised with a dictionary mapping parameter names to continuous numeric intervals or discrete sets.
The former are given as python :obj:`list` and the latter---as :obj:`set`.

For example, to define a domain over the continuous interval [-10, 10] and the discrete set of
strings {"option_1", "option_2"}, it suffices to write:

.. code-block:: python

    domain = Domain({"var_1": [-10, 10], "var_2": {"option_1", "option_2"}})

where ``"var_1"`` and ``"var_2"`` are two arbitrary names for the two subdomains.

Given this domain we can now generate samples from it using the :py:meth:`sample()` method:

.. code-block:: python

    >>> domain.sample()
    {'var_1': -8.529187978165552, 'var_2': 'option_1'}

The returned objects are of class :class:`Sample` and represent one realisation of the domain.
It is represented as a mapping of parameter names to samples from the set of possible values.
It also has a handy conversion methods such as :py:meth:`as_dict()` or :py:meth:`as_namedtuple()` which enable accessing
parameters using the `["var_1"]` or `.var_1` notation.

Both :class:`Domain` and :class:`Sample` objects allow for nested subdomains, e.g.:

.. code-block:: python

    >>> domain = Domain({
    ...    "subdomain_a": {"var_1": [-10, 10], "var_2": {"option_1", "option_2"}},
    ...    "subdomain_b": {"var_1": [-1, 1], "var_2": {"option_1", "option_2"}}
    ... })
    >>> sample = domain.sample()
    >>> sample
    {
        'subdomain_a': {'var_1': -6.892359956494582, 'var_2': 'option_2'},
        'subdomain_b': {'var_1': 0.21004903180560652, 'var_2': 'option_1'}
    }
    >>> nt_sample = sample.as_namedtuple()
    >>> nt_sample.subdomain_a.var_2
    'option_2'
