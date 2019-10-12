Reports
=======

Saving and visualising progress can be accomplished by using :class:`Reporter` instance.
The reporter is supplied with data using the :py:meth:`log()` method which takes a tuple of a sample and score.
Optionally one can store additional information about the current experiment, e.g. the output directory or the job id,
using the ``meta`` keyword argument:

.. code-block:: python

    for s, e, m in zip(samples, evaluations, meta_infos):
        reporter.log((s, e), meta=m)

Table
-----

Hypertunity comes with a built-in reporter which organises the experiment results into an ascii table.
It is initialised from a domain and a list of metrics and can be viewed as a formatted string table by calling :obj:`str`
on the object.
The table can be sorted in ascending or descending order and the best results can be emphasised:

.. code-block:: python

    >>> domain = ht.Domain({"x": [-5., 6.], "y": {"sin", "cos"}, "z": set(range(4))})
    >>> reporter = ht.Table(domain, metrics=["score"])
    >>> # run experiment and call reporter.log(...)
    ...
    >>> print(reporter.format(order="descending"))
    +=====+========+=====+===+==============+
    | No. |   x    |  y  | z |    score     |
    +=====+========+=====+===+==============+
    |  6  | -4.35  | cos | 1 | 15.921 ± 0.0 |
    +-----+--------+-----+---+--------------+
    |  5  | -4.232 | cos | 3 | 8.906 ± 0.0  |
    +-----+--------+-----+---+--------------+
    |  4  | -4.588 | sin | 3 | 6.134 ± 0.0  |
    +-----+--------+-----+---+--------------+
    |  2  |  2.16  | cos | 0 | 4.667 ± 0.0  |
    +-----+--------+-----+---+--------------+
    |  3  | -0.977 | cos | 1 | -2.045 ± 0.0 |
    +-----+--------+-----+---+--------------+
    |  1  | -1.438 | cos | 3 | -6.933 ± 0.0 |
    +-----+--------+-----+---+--------------+

Tensorboard
-----------

If Hypertunity is installed with the `tensorboard` option, a suitable version of Tensorflow and Tensorboard will be installed.
This will enable a :class:`Tensorboard` reporter which, using the HParams plugin, will generate live visualisations
as experiments are being logged. One can start the Tensorboard dashboard in the browser as usual, using the `logdir` supplied
at initialisation.

Note that to create a Tensorboard reporter one will have to import ``hypertunity.reports.tensorboard`` explicitly:

.. code-block:: python

    import hypertunity.reports.tensorboard as tb
    tb_reporter = tb.Tensorboard(domain, metrics=["score"], logdir="./logs")

See the :doc:`quickstart` guide for a preview of the dashboard visualisation.
