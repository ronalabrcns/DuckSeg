============
Contributing
============

Bug reports and pull requests are welcome on
`GitHub <https://github.com/ronalabrcns/DuckSeg/issues>`_.

Development setup
=================

.. code-block:: bash

   git clone https://github.com/ronalabrcns/DuckSeg.git
   cd DuckSeg
   pip install -e ".[docs]"
   pip install git+https://github.com/vanvalenlab/cellSAM.git

Documentation
=============

The API reference is generated from the docstrings, so the docstrings are
the documentation. Please

* write `NumPy-style <https://numpydoc.readthedocs.io/en/latest/format.html>`_
  docstrings (``Parameters``, ``Returns``, ``Raises``, ``Examples``); the
  Google style (``Args:``) is not rendered,
* start with a one-line summary and explain *what it means for the
  experiment*, not only what the code does,
* state the units and conventions of array arguments (rows are the first
  axis; the ROI ranges are row indices),
* add a new module to ``docs/api/`` and to ``docs/modules.rst``.

Build the site locally with warnings treated as errors; Read the Docs
does the same:

.. code-block:: bash

   sphinx-build -W --keep-going -b html docs docs/_build/html

Extending DuckSeg
=================

* A new preprocessing step is a function from frame to frame
  (:doc:`guides/preprocessing`).
* A new quality filter is a factory that returns a function from
  :class:`~DuckSeg.experiment_evaluator.CellVideo` to ``bool``
  (:doc:`guides/outlier_filtering`); add it to
  ``src/DuckSeg/outlier_filters.py``.
