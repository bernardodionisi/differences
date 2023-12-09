Unit Tests
==========

The :obj:`differences` library uses `pytest <https://docs.pytest.org/en/7.0.x/>`_ for unit testing.

Install
-------

First, `pytest` needs to be installed on your system. Easily install it by installing the development dependencies.

.. code-block:: console

   $ python3 -m pip install .[dev]

Configuration
-------------

The `pytest` configuration is stored in `pyproject.toml`.

.. literalinclude:: ../../pyproject.toml
   :caption: pyproject.toml
   :start-at: [tool.pytest.ini_options]
   :end-before: [tool.coverage.report]
   :language: toml

Run from the command line
-------------------------

Execute all of the unit tests manually from the command line.

.. code-block:: console

    $ python3 -m pytest tests/

Or only run a specific test file.

.. code-block:: console

    $ python3 -m pytest tests/test_*.py

Or only run a specific unit test.

.. code-block:: console

    $ python3 -m pytest tests/test_*.py::test_*