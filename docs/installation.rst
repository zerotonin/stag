Installation
============

Requirements
------------

- Python ≥ 3.10
- A CUDA-capable GPU is recommended for clustering (Stage 4) but not
  required — the pipeline falls back to scikit-learn on CPU.

With conda (recommended)
------------------------

The conda environment includes RAPIDS cuML for GPU-accelerated *k*-means:

.. code-block:: bash

   git clone https://github.com/zerotonin/stag.git
   cd stag
   conda env create -f environment.yml
   conda activate stag
   pip install -e .

With pip (CPU only)
-------------------

.. code-block:: bash

   git clone https://github.com/zerotonin/stag.git
   cd stag
   pip install -e .

To include GPS trajectory support:

.. code-block:: bash

   pip install -e ".[gps]"

To install documentation and development tools:

.. code-block:: bash

   pip install -e ".[docs,dev]"

Verifying the installation
--------------------------

.. code-block:: python

   import stag
   print(stag.__version__)  # Should print 0.1.0

Building the documentation
--------------------------

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.
