"""
Module to install r-packages
"""
from __future__ import annotations

import argparse

import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector  # R vector of strings

required = [
    "did",
]
# ----------------------------------------------------------------------

# import R's utility package
r_utils = rpackages.importr("utils")

# select a mirror for R packages
r_utils.chooseCRANmirror(ind=1)  # select the first mirror in the list


def install_rpackages(r_pkgs: list = None):
    # Selectively install what needs to be installed.

    if r_pkgs is None:
        r_pkgs = required
    else:
        r_pkgs += required

    if not r_pkgs:
        print("no library specified")

    else:
        names_to_install = [x for x in set(r_pkgs) if not rpackages.isinstalled(x)]

        if names_to_install:
            names = ", ".join(names_to_install)
            print(f"the following R libraries will be installed: {names}")
            r_utils.install_packages(StrVector(names_to_install))

        else:
            print("all libraries already installed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-l",
        "--list",
        nargs="+",
        required=False,
        help="A list or R packages to install from CRAN",
    )
    args = parser.parse_args()

    install_rpackages(r_pkgs=args.list)
