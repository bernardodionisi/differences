from __future__ import annotations

from pathlib import Path

import polars as pl


def _read_data(file_path: Path, **kwargs) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    suffix = Path(file_path).suffix

    if suffix == ".parquet":
        return pl.read_parquet(file_path, **kwargs)

    elif suffix == ".csv":
        return pl.read_csv(file_path, **kwargs)

    elif suffix in [".dta"]:
        return _read_data_pandas(file_path=file_path, **kwargs)

    else:
        # You can add more file formats here
        raise ValueError(f"Unsupported file format {suffix}")


def _scan_data(file_path: Path, **kwargs) -> pl.LazyFrame:
    suffix = Path(file_path).suffix

    if suffix == ".parquet":
        return pl.scan_parquet(file_path, **kwargs)

    elif suffix == ".csv":
        return pl.scan_csv(file_path, **kwargs)

    else:
        raise ValueError(f"Unsupported file format {suffix}")


def _read_data_pandas(file_path: Path, **kwargs) -> pl.DataFrame | tuple[pl.DataFrame, dict]:
    suffix = Path(file_path).suffix

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(f"You need to install pandas to read a '{suffix}' file.")

    if suffix in ".dta":
        data = pd.read_stata(file_path, **kwargs)

        _info = {}
        with pd.io.stata.StataReader(file_path) as reader:
            # data_label: str = reader.data_label
            # value_labels: dict = reader.value_labels()
            variable_labels: dict = reader.variable_labels()

        _info["variables_info"] = variable_labels

        return pl.from_pandas(data), _info

    else:
        raise ValueError(f"Unsupported file format {suffix}")

# ======================================================================


# @classmethod
# def _input_from_file(cls, file_path: Path, scan: bool = False, **kwargs) -> Self:
#     file_path = Path(file_path)
#
#     info = {
#         "file_info": {"file_path": str(file_path)}
#     }
#
#     if scan:
#         data = _scan_data(file_path, **kwargs)
#     else:
#         data = _read_data(file_path, **kwargs)
#
#     if isinstance(data, tuple):
#         data, _info = data
#         info.update(_info)
#     # TODO: wrong INFO management, here its the old one
#     # return cls(data=data, name=file_path.stem).info_update(**info)
#
# @classmethod
# def read(cls, file_path: str | Path, **kwargs) -> Self:
#     """
#     Utility method to read data from file_path
#
#     Group
#     -----
#         Constructors
#
#     Parameters
#     ----------
#     file_path
#     kwargs
#
#     Returns
#     -------
#
#     """
#     return cls._input_from_file(file_path=file_path, scan=False, **kwargs)
#
# @classmethod
# def scan(cls, file_path: str | Path, **kwargs) -> Self:
#     """
#     Utility method to scan data from file_path
#
#     Group
#     -----
#         Constructors
#
#     Parameters
#     ----------
#     file_path
#     kwargs
#
#     Returns
#     -------
#
#     """
#     return cls._input_from_file(file_path=file_path, scan=True, **kwargs)
