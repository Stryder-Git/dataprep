import pytest
import pandas as pd
from dataprep import Data
from dask.delayed import delayed

@delayed
def delayer(x): return x

@pytest.mark.parametrize("data", [
    dict(A=pd.Index([1, 2, 3]), B=pd.DatetimeIndex(["2000-01-01", "2000-01-02"]))
])
def test_compute(data):
    d = Data({s: delayer(d) for s, d in data.items()})

    result = d.compute()
    assert result == data

