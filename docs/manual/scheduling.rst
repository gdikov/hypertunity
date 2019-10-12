Scheduling jobs
===============

Often in practice the objective function is a python script that might take command line arguments as parameters or define a function that has lots of dependencies.
Importing this function into the hyperparameter optimisation script or wrapping the target script involves some boilerplate code.
To help with that Hypertunity allows for specifying objective functions as ``Job`` instances which are then run in succession or in parallel using a ``Scheduler``.
The latter is a wrapper around `joblib <https://joblib.readthedocs.io>`_ and takes care of both running jobs and collecting results.

Scheduling of ``Job`` instances is done using the ``dispatch`` method of a ``Scheduler``:

.. code-block:: python

    jobs = [Job(...) for _ in range(10)]
    scheduler.dispatch(jobs)
    evaluations = [r.data for r in scheduler.collect(n_results=batch_size, timeout=10.0)]

There are multiple ways to define a job depending on the target to optimise.

Local python callable
~~~~~~~~~~~~~~~~~~~~~

If the function is defined or imported within the hyperparameter optimisation script, the ``task`` argument is the callable instance.
The ``args`` is then a tuple of arguments or a dict of named arguments which are supplied to the task function during calling.
For example:

.. code-block:: python

    jobs = [ht.Job(task=foo, args=(*s.as_namedtuple(),)) for s in samples]


Python callable in a script
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the function to optimise resides in a script, Hypertunity allows for specifying a target by the full path to the script.
To select the objective function from the script append ``:`` and the function name:

.. code-block:: python

    jobs = [Job(task="path/to/script.py:foo", args=(*s.as_namedtuple(),)) for s in samples]


A script
~~~~~~~~

If the objective function is a full command line application or a script that accepts the hyperparameters to tune as command line arguments then you should create a job as follows:

.. code-block:: python

    jobs = [Job(task="path/to/script.py",
                args=(*s.as_namedtuple(),),
                meta={"binary": "python"}) for s in samples]


Using Slurm
~~~~~~~~~~~

To schedule jobs using Slurm a special job type is available. It allows to configure resources and other Slurm parameters but also requires that the target script is able to write a results file on disk.

.. code-block:: python

    jobs = [SlurmJob(task="path/to/script.py",
                     args=(*sample.as_namedtuple(),),
                     output_file="path/to/results.pkl",
                     meta={"binary": "python", "resources": {"cpu": 1}}))

