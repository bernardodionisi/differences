"""Local CI parity: mirrors .github/workflows for pre-push checks."""

import nox

PYTHON_DEFAULT = "3.12"


@nox.session(python=PYTHON_DEFAULT)
def tests(session):
    session.install("-e", ".[test]")
    session.run("pytest", "-n", "auto", "tests/")


@nox.session(python=PYTHON_DEFAULT)
def docs(session):
    session.install("-e", ".[docs,all]")
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "--keep-going",
        "docs/",
        "docs/build/html",
    )
