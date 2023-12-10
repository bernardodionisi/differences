from __future__ import annotations

import sys

try:
    from differences._version import __version__, __version_tuple__
except ModuleNotFoundError:  # pragma: no cover
    import warnings

    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
    warnings.warn(
        "\nAn error occurred during package install "
        "where setuptools_scm failed to create a _version.py file."
        "\nDefaulting version to 0.0.0."
    )


def show_versions() -> None:
    r"""
    Print out version of Differences and dependencies to stdout.

    """  # noqa: W505
    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    deps = _get_dependency_info()
    core_properties = ("differences", "Platform", "Python")
    keylen = max(len(x) for x in [*core_properties, *deps.keys()]) + 1

    print("-------- Version info ---------")
    print(f"{'Platform:':{keylen}s} {platform.platform()}")
    print(f"{'Python:':{keylen}s} {sys.version}\n")

    print(f"{'differences:':{keylen}s} {__version__}")

    print("\n---- Optional dependencies ----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")


def _get_dependency_info() -> dict[str, str]:
    # from pyproject.toml [all]
    opt_deps = [
        "numpy",
        "scipy",
        "pandas",
        "formulaic",
        "linearmodels",
        "statsmodels",
        "tqdm",
        "joblib",
        "plotto",
    ]
    return {f"{name}:": _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    # note: we import 'importlib' here as a significiant optimisation for initial import
    import importlib
    import importlib.metadata

    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        module_version = importlib.metadata.version(dep_name)  # pragma: no cover

    return module_version


# ---------------------------- docs ------------------------------------


def show_versions_docs() -> None:
    r"""
    Print out version of Differences and Docs dependencies to stdout.
    """  # noqa: W505
    # note: we import 'platform' here as a micro-optimisation for initial import
    import platform

    deps = _get_dependency_info_docs()
    core_properties = ("differences",)
    keylen = max(len(x) for x in [*core_properties, *deps.keys()]) + 1

    print("-------- Version info ---------")
    print(f"{'differences:':{keylen}s} {__version__}")

    print("\n---- Documentation dependencies ----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")


def _get_dependency_info_docs() -> dict[str, str]:
    # from pyproject.toml [dpcs]
    docs_deps = [
        "sphinx",
        "sphinx_immaterial",
        "sphinx_design",
        "sphinx_last_updated_by_git",
        "ipykernel",
    ]
    return {f"{name}:": _get_dependency_version(name) for name in docs_deps}
