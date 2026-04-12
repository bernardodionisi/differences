Installation
============

Install from PyPI
-----------------

The latest released version of :obj:`differences` can be installed from `PyPI <https://pypi.org/project/differences/>`_ using `pip`. Requires Python >= 3.10.

.. code-block:: console

   $ pip install differences


Optional dependencies
---------------------

The ``[all]`` extra pulls in :obj:`linearmodels`, which is needed by :func:`differences.generate_data`:

.. code-block:: console

   $ pip install "differences[all]"


Install from GitHub
-------------------

To install the latest development version directly from a branch:

.. code-block:: console

   $ pip install "git+https://github.com/bernardodionisi/differences.git@main"


Development install
-------------------

For local development, clone the repo and install in editable mode with the test extras:

.. code-block:: console

   $ git clone https://github.com/bernardodionisi/differences.git
   $ cd differences
   $ pip install -e ".[all,test]"

Optionally, enable the repo's pre-push hook to run the CI checks (``nox``) before each push:

.. code-block:: console

   $ git config core.hooksPath .githooks


Import
------

Import the :obj:`differences` package in Python.

.. ipython:: python

   import differences
   differences.__version__


