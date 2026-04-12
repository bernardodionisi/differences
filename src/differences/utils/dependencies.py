# This module has been adapted from:
#     https://github.com/pola-rs/polars/blob/main/py-polars/polars/dependencies.py
#
# py-polars/polars/dependencies.py is distributed with the following license
#
# '''
# Copyright (c) 2025 Ritchie Vink
# Some portions Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# '''

from __future__ import annotations

import re
import sys
from functools import cache
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Hashable

_LINEARMODELS_AVAILABLE = True


class _LazyModule(ModuleType):
    """
    Module that can act both as a lazy-loader and as a proxy.

    Notes
    -----
    We do NOT register this module with `sys.modules` so as not to cause
    confusion in the global environment. This way we have a valid proxy
    module for our own use, but it lives *exclusively* within polars.

    """

    __lazy__ = True

    _mod_pfx: ClassVar[dict[str, str]] = {
        "linearmodels": "lm.",
    }

    def __init__(
            self,
            module_name: str,
            *,
            module_available: bool,
    ) -> None:
        """
        Initialise lazy-loading proxy module.

        Parameters
        ----------
        module_name : str
            the name of the module to lazy-load (if available).

        module_available : bool
            indicate if the referenced module is actually available (we will proxy it
            in both cases, but raise a helpful error when invoked if it doesn't exist).

        """
        self._module_available = module_available
        self._module_name = module_name
        self._globals = globals()
        super().__init__(module_name)

    def _import(self) -> ModuleType:
        # import the referenced module, replacing the proxy in this module's globals
        module = import_module(self.__name__)
        self._globals[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, attr: Any) -> Any:
        # have "hasattr('__wrapped__')" return False without triggering import
        # (it's for decorators, not modules, but keeps "make doctest" happy)
        if attr == "__wrapped__":
            msg = f"{self._module_name!r} object has no attribute {attr!r}"
            raise AttributeError(msg)

        # accessing the proxy module's attributes triggers import of the real thing
        if self._module_available:
            # import the module and return the requested attribute
            module = self._import()
            return getattr(module, attr)

        # user has not installed the proxied/lazy module
        elif attr == "__name__":
            return self._module_name
        elif re.match(r"^__\w+__$", attr) and attr != "__version__":
            # allow some minimal introspection on private module
            # attrs to avoid unnecessary error-handling elsewhere
            return None
        else:
            # all other attribute access raises a helpful exception
            pfx = self._mod_pfx.get(self._module_name, "")
            msg = f"{pfx}{attr} requires {self._module_name!r} module to be installed"
            raise ModuleNotFoundError(msg) from None


def _lazy_import(module_name: str) -> tuple[ModuleType, bool]:
    """
    Lazy import the given module; avoids up-front import costs.

    Parameters
    ----------
    module_name : str
        name of the module to import, eg: "pyarrow".

    Notes
    -----
    If the requested module is not available (eg: has not been installed), a proxy
    module is created in its place, which raises an exception on any attribute
    access. This allows for import and use as normal, without requiring explicit
    guard conditions - if the module is never used, no exception occurs; if it
    is, then a helpful exception is raised.

    Returns
    -------
    tuple of (Module, bool)
        A lazy-loading module and a boolean indicating if the requested/underlying
        module exists (if not, the returned module is a proxy).

    """
    # check if module is LOADED
    if module_name in sys.modules:
        return sys.modules[module_name], True

    # check if module is AVAILABLE
    try:
        module_spec = find_spec(module_name)
        module_available = not (
                module_spec is None or module_spec.loader is None
        )
    except ModuleNotFoundError:
        module_available = False

    # create lazy/proxy module that imports the real one on first use
    # (or raises an explanatory ModuleNotFoundError if not available)
    return (
        _LazyModule(
            module_name=module_name,
            module_available=module_available,
        ),
        module_available,
    )


if TYPE_CHECKING:
    import copy
    import dataclasses
    import decimal
    import functools
    import hashlib
    import inspect
    import io
    import itertools
    import json
    import pathlib
    import re
    import shlex

    import linearmodels

else:
    # infrequently-used builtins
    io, _ = _lazy_import("io")
    json, _ = _lazy_import("json")
    copy, _ = _lazy_import("copy")
    re, _ = _lazy_import("re")
    itertools, _ = _lazy_import("itertools")
    collections, _ = _lazy_import("collections")
    inspect, _ = _lazy_import("inspect")
    dataclasses, _ = _lazy_import("dataclasses")
    functools, _ = _lazy_import("functools")
    decimal, _ = _lazy_import("decimal")
    hashlib, _ = _lazy_import("hashlib")
    pathlib, _ = _lazy_import("pathlib")
    shlex, _ = _lazy_import("shlex")

    # optional third party libs
    linearmodels, _LINEARMODELS_AVAILABLE = _lazy_import("linearmodels")


@cache
def _might_be(cls: type, type_: str) -> bool:
    # infer whether the given class "might" be associated with the given
    # module (in which case it's reasonable to do a real isinstance check)
    try:
        return any(f"{type_}." in str(o) for o in cls.mro())
    except TypeError:
        return False


def _check_for_linearmodels(obj: Any, *, check_type: bool = True) -> bool:
    return _LINEARMODELS_AVAILABLE and _might_be(
        cast("Hashable", type(obj) if check_type else obj), "linearmodels"
    )


__all__ = [
    # lazy-load rarely-used/heavy builtins
    "re",
    "json",
    "copy",
    "itertools",
    "inspect",
    "collections",
    "dataclasses",
    "functools",
    "decimal",
    "io",
    "hashlib",
    "pathlib",
    "shlex",
    # lazy-load third party libs
    "linearmodels",
    # lazy utilities
    "_check_for_linearmodels",
    "_LazyModule",
    # exported flags/guards
    "_LINEARMODELS_AVAILABLE",
]
