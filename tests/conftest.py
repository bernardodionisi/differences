from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def mpdta():
    path = Path(__file__).resolve().parents[1] / "data" / "mpdta.parquet"
    df = pd.read_parquet(path)
    return df.set_index(["countyreal", "year"])
