Contributing
============

We welcome contributions to STAG. Here are some guidelines.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/zerotonin/stag.git
   cd stag
   conda env create -f environment.yml
   conda activate stag
   pip install -e ".[dev,docs]"

Code style
----------

We use `ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting,
configured in ``pyproject.toml``. Docstrings follow the
`NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

.. code-block:: bash

   ruff check stag/          # lint
   ruff format stag/         # format

Running tests
-------------

.. code-block:: bash

   pytest tests/ -v

Pull requests
-------------

1. Fork the repository and create a feature branch.
2. Add tests for new functionality.
3. Ensure ``ruff check`` and ``pytest`` pass.
4. Submit a pull request with a clear description.

Reporting issues
----------------

Please open a GitHub issue with a minimal reproducible example and the
output of ``python -c "import stag; print(stag.__version__)"``.
