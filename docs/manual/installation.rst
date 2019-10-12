Installation
============

Requirements
------------

Hypertunity has been tested with Python 3.6 and 3.7. As of now, there are no plans to support earlier versions of Python.
The reason for that is the usage of variable and function annotations, dataclasses as well as relying on the fact that the
insertion order of the keys in a dictionary is preserved during iteration. Porting Hypertunity to earlier versions will
only make it unnecessarily hard to maintain.

From PyPI
---------

To get the latest stable release just run:

.. code-block:: bash

    pip install hypertunity

Note that this will install the basic version only, without support for Tensorboard visualisations.
To enable this feature you will need to specify the option `tensorboard`.
To run the tests or compile the docs add the `tests` and `docs` options respectively:

.. code-block:: bash

    pip install hypertunity[tensorboard,tests,docs]


From source
-----------

To install the bleeding-edge version of Hypertunity, clone the repository, checkout the master branch
and install from source:

.. code-block:: bash

    git clone https://github.com/gdikov/hypertunity.git
    cd hypertunity
    git checkout master
    pip install ./[tensorboard,tests,docs]
