from __future__ import annotations

try:
    from ._version import __version__, __version_tuple__
except ModuleNotFoundError:  # pragma: no cover
    import warnings

    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
    warnings.warn(
        "\nAn error occurred during package install "
        "where setuptools_scm failed to create a _version.py file."
        "\nDefaulting version to 0.0.0."
    )

from differences.models.did.attgt.attgt import ATTgt

from differences.models.lm.twfe import TWFE
from differences.models.lm.panel_models import PanelModels
from differences.models.lm.panel_ols import panel_ols_fit
