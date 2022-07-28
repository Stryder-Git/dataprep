import pandas as pd
from pandas.testing import assert_series_equal
import pytest

import dataprep as dp
import dp_tests.utils as u

ix = pd.date_range("2000-01-01", periods= 5)

datafs = dict(A=pd.DataFrame(dict(a=range(5), b=range(5)), index=ix),
              B=pd.DataFrame(dict(a=range(5), b=range(5)), index=ix))

def test_from_pandas_df():
    data = dp.from_pandas(datafs)
    assert isinstance(data, dp.DataSet)
    assert data.all_columns_equal
    assert data.all_columns_equal

    orig_names = {k: d.columns for k, d in datafs.items()}
    names = {k: d.columns for k, d in datafs.items()}
    u.dict_same(orig_names, names)



index = dict(A= ix, B= ix)

series = dict(A = pd.Series(range(5), index= ix),
              B = pd.Series(range(5), index= ix))

seriesn = {k: pd.Series(s, name= "name") for k, s in series.items()}

@pytest.mark.parametrize("data, name, result", [
    (index, None, dict(A= 0, B= 0)),
    (index, "dtindex", dict(A= "dtindex", B= "dtindex")),

    (series, None, dict(A= 0, B= 0)),
    (series, "other", dict(A="other", B="other")),

    (seriesn, None, dict(A= "name", B= "name")),
    (seriesn, "other", dict(A= "other", B= "other")),
])
def test_from_pandas_other(data, name, result):

    result = pd.Series({k: [v] for k, v in result.items()})
    data = dp.from_pandas(data, name= name)

    assert isinstance(data, dp.DataSet)
    print(data.all_columns, result)
    assert_series_equal(data.all_columns, result)


@pytest.mark.parametrize("files, name, dtindex", [
    ("dp_tests\\sample_data", lambda p: p.split(".")[0], None),
    (["dp_tests\\sample_data\\A.csv", "dp_tests\\sample_data\\B.csv"], lambda p: p.split(".")[0][-1], None),

    ("dp_tests\\sample_data", lambda p: p.split(".")[0], "Unnamed: 0"),
    (["dp_tests\\sample_data\\A.csv", "dp_tests\\sample_data\\B.csv"], lambda p: p.split(".")[0][-1], "Unnamed: 0"),
])

def test_from_files(files, name, dtindex):
    data = dp.from_files(files, name, dtindex= dtindex)

    assert (data.symbols == ["A", "B"]).all()

    columns = ["a", "b"]
    noix = dtindex is None
    if noix: columns.insert(0, "Unnamed: 0")

    assert data.all_columns_equal
    assert (data.common_columns == columns).all()

    if noix: return
    ixdtype = data.apply(lambda d: d.index.dtype.name).compute()
    ixdtype  == {"A": "datetime64[ns]", "B": "datetime64[ns]"}











