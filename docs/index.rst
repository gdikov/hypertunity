:github_url: https://github.com/gdikov/hypertunity

.. image:: _static/images/logo.svg
  :width: 800
  :align: center
  :alt: Hypertunity logo

========
Welcome!
========

Hypertunity is a lightweight, high-level library for hyperparameter optimisation.
Among others, it supports:

* Bayesian optimisation by wrapping `GPyOpt <http://sheffieldml.github.io/GPyOpt/>`_
* external or internal objective evaluation using a scheduler, also compatible with `Slurm <https://slurm.schedmd.com>`_
* real-time visualisation of results in `Tensorboard <https://www.tensorflow.org/tensorboard>`_ using the `HParams <https://www.tensorflow.org/tensorboard/r2/hyperparameter_tuning_with_hparams>`_ plugin.

The main guiding design principles are:

* **Modular**: you can use any optimiser and reporter as well as schedule jobs locally or on Slurm without changes in the API.
* **Simple**: the small codebase (just about 1000 LOC) and the flat subpackage hierarchy makes it easy to use, maintain and extend.
* **Extensible**: base classes such as :class:`Optimiser`, :class:`Job` and :class:`Reporter` allow for seamless implementation of customized functionality.


.. toctree::
  :maxdepth: 2
  :caption: User Guide

  manual/installation
  manual/quickstart
  manual/domain
  manual/optimisation
  manual/reports
  manual/scheduling


.. toctree::
  :maxdepth: 2
  :caption: API Reference

  source/hypertunity
  source/optimisation
  source/reports
  source/scheduling


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
