import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
import pandas_market_calendars as mcal
from dataprep import Data, DataSet

import dp_tests.utils as u


"""
Permutations

    columns that are not in all

    indexes covering much less data

    indexes that have missing rows

    indexes that have missing sessions

        include breaks
    

TODO:
    
    The fake opens and closes in the index below messes with missing_sessions and incomplete_sessions tests


"""

npr = np.random.default_rng(1)

# The following index is based on pandas_market_calendars.get_calendar("XHKG").schedule, which has a break
# between 4 and 5 am UTC. Although, special opens/closes on Jan 4th and Jan 6th are faked for testing.
ix = pd.DatetimeIndex(['2000-01-03 02:00:00', '2000-01-03 03:30:00', '2000-01-03 05:00:00', '2000-01-03 06:30:00',
                       '2000-01-04 03:00:00', '2000-01-04 05:00:00', '2000-01-04 06:30:00',  # fake late open
                       '2000-01-05 02:00:00', '2000-01-05 03:30:00', '2000-01-05 05:00:00', '2000-01-05 06:30:00',
                       '2000-01-06 02:00:00', '2000-01-06 03:30:00',  # fake early close
                       '2000-01-07 02:00:00', '2000-01-07 03:30:00', '2000-01-07 05:00:00', '2000-01-07 06:30:00'],
                      tz='UTC')

def _drop(frm, ixs): return frm[~np.isin(np.arange(frm.shape[0]), ixs)]

ln = ix.shape[0]
price_data = dict(
    A=pd.DataFrame(npr.integers(100, size=(ln, 4)), index=ix, columns=list("abcd")), # complete
    B=pd.DataFrame(npr.integers(100, size=(ln, 3)), index=ix, columns=list("abd")), # not all columns
    C=pd.DataFrame(npr.integers(100, size=(ln-4, 4)), index=ix[:-4], columns=list("abcd")), # doesn't cover whole range
    D=pd.DataFrame(npr.integers(100, size=(ln-4, 4)), index=ix[4:], columns=list("abcd")), # doesn't cover whole range

    # With missing indexes
    E=pd.DataFrame(npr.integers(100, size=(ln-4, 4)), columns=list("abcd"),
                   index= _drop(ix, [0, 5, 10, 16]))
)

test_data = [price_data,]

@pytest.mark.parametrize("data", test_data)
def test_column_properties(data):
    ds = DataSet(data)

    assert ds.all_columns_equal is False
    assert len(ds.common_columns) != ds.all_columns.str.len().max()

    ds = ds[ds.common_columns]
    assert ds.all_columns_equal


@pytest.mark.parametrize("data", test_data)
def test_index_properties(data):

    ds = DataSet(data)
    assert ds.all_indexes_equal is False

    assert_frame_equal(ds.index_ranges,
                       pd.DataFrame({
                           0: pd.Series(['2000-01-03 02:00:00', '2000-01-03 02:00:00', '2000-01-03 02:00:00',
                                         '2000-01-04 03:00:00', '2000-01-03 03:30:00'],
                                        dtype='datetime64[ns]').dt.tz_localize('UTC'),
                           1: pd.Series(['2000-01-07 06:30:00', '2000-01-07 06:30:00', '2000-01-06 03:30:00',
                                         '2000-01-07 06:30:00', '2000-01-07 05:00:00'],
                                        dtype='datetime64[ns]').dt.tz_localize('UTC')
                       }).set_index(pd.Index(['A', 'B', 'C', 'D', 'E'])))

    ds = ds.select(list("AB"))
    assert ds.all_indexes_equal


@pytest.mark.parametrize("data", test_data)
def test_dataset_properties(data):
    ds = DataSet(data)

    commons = len(ds.common_columns)
    assert ds.shape == (len(data), commons)
    assert ds.fshape == (sum((x.shape[0] for x in data.values())), commons)

    assert_frame_equal(ds.shapes, pd.DataFrame({
                        0: pd.Series([17, 17, 13, 13, 13]),
                        1: pd.Series([4, 3, 4, 4, 4])
                        }).set_index(pd.Index(['A', 'B', 'C', 'D', 'E'])))


    assert list(ds.symbols) == list(data.keys())

sched = pd.DataFrame(dict(
    market_open = ['2000-01-03 02:00:00', '2000-01-04 03:00:00', '2000-01-05 02:00:00', '2000-01-06 02:00:00', '2000-01-07 02:00:00'],
    break_start = ['2000-01-03 04:00:00', '2000-01-04 04:00:00', '2000-01-05 04:00:00', '2000-01-06 04:00:00', '2000-01-07 04:00:00'],
    break_end = ['2000-01-03 05:00:00', '2000-01-04 05:00:00', '2000-01-05 05:00:00', '2000-01-06 04:00:00', '2000-01-07 05:00:00'],
    market_close = ['2000-01-03 08:00:00', '2000-01-04 08:00:00', '2000-01-05 08:00:00', '2000-01-06 04:00:00', '2000-01-07 08:00:00'],
), dtype= "datetime64[ns, UTC]")

@pytest.mark.parametrize("data, expected", [
    (dict(
    A = pd.DataFrame(index= ix[ix.normalize()!=pd.Timestamp("2000-01-06", tz= "UTC")]),
    B = pd.DataFrame(index= ix[ix.normalize()!=pd.Timestamp("2000-01-03", tz= "UTC")]), # shouldn't be considered missing (because it's the first date)
    C = pd.DataFrame(index= ix[ix.normalize()!=pd.Timestamp("2000-01-05", tz= "UTC")])
    ), {'A': pd.DatetimeIndex(['2000-01-06 02:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None),
        'B': pd.DatetimeIndex([], dtype='datetime64[ns, UTC]', freq=None),
        'C': pd.DatetimeIndex(['2000-01-05 02:00:00+00:00', '2000-01-05 05:00:00+00:00'], dtype='datetime64[ns, UTC]', freq=None)})
])
def test_missing_sessions(data, expected):
    ds = DataSet(data)
    missing = ds.missing_sessions(sched, "1.5H")

    assert isinstance(missing, Data) and not isinstance(missing, DataSet)
    assert u.dict_same(missing.compute(), expected)


incomplete_data = dict(
    A = pd.DataFrame(index= ix[3:]), # missing '2000-01-03 02:00:00', '2000-01-03 03:30:00', '2000-01-03 05:00:00'
    B = pd.DataFrame(index= _drop(ix, [5])), # missing '2000-01-04 05:00:00'
    C = pd.DataFrame(index= _drop(ix, [7, 8])), # missing '2000-01-05 02:00:00', '2000-01-05 03:30:00'
    D = pd.DataFrame(index= ix[:-2]), # missing '2000-01-07 05:00:00', '2000-01-07 06:30:00'
    E = pd.DataFrame(index= ix)) # missing []

@pytest.mark.parametrize("data, expected", [
    (incomplete_data
    ,dict(
        A = pd.DatetimeIndex(['2000-01-03 05:00:00'], dtype= "datetime64[ns, UTC]", freq= None),
        B = pd.DatetimeIndex(['2000-01-04 05:00:00'], dtype= "datetime64[ns, UTC]", freq= None),
        C = pd.DatetimeIndex([], dtype= "datetime64[ns, UTC]", freq= None),
        D = pd.DatetimeIndex([], dtype= "datetime64[ns, UTC]", freq= None),
        E = pd.DatetimeIndex([], dtype= "datetime64[ns, UTC]", freq= None),
    ))])
def test_incomplete_sessions(data, expected):
    ds = DataSet(data)
    incomplete = ds.incomplete_sessions(sched, "1.5H")

    assert isinstance(incomplete, Data) and not isinstance(incomplete, DataSet)
    assert u.dict_same(incomplete.compute(), expected)


@pytest.mark.parametrize("data, expected", [
    (incomplete_data,
     dict(
        A = pd.DatetimeIndex(['2000-01-03 02:00:00', '2000-01-03 05:00:00'], dtype= "datetime64[ns, UTC]", freq= None),
        B = pd.DatetimeIndex(['2000-01-04 05:00:00'], dtype= "datetime64[ns, UTC]", freq= None),
        C = pd.DatetimeIndex(['2000-01-05 02:00:00'], dtype= "datetime64[ns, UTC]", freq= None),
        D = pd.DatetimeIndex(['2000-01-07 05:00:00'], dtype= "datetime64[ns, UTC]", freq= None),
        E = pd.DatetimeIndex([], dtype= "datetime64[ns, UTC]", freq= None)))
])
def test_incomplete_or_missing_sessions(data, expected):
    ds = DataSet(data)
    incomp_miss = ds.incomplete_or_missing_sessions(sched, "1.5H")

    assert isinstance(incomp_miss, Data) and not isinstance(incomp_miss, DataSet)
    assert u.dict_same(incomp_miss.compute(), expected)


@pytest.mark.parametrize("data, expected", [
    (incomplete_data,
     dict(
         A=pd.DatetimeIndex(['2000-01-03 02:00:00', '2000-01-03 03:30:00', '2000-01-03 05:00:00'], dtype="datetime64[ns, UTC]", freq=None),
         B=pd.DatetimeIndex(['2000-01-04 05:00:00'], dtype="datetime64[ns, UTC]", freq=None),
         C=pd.DatetimeIndex(['2000-01-05 02:00:00', '2000-01-05 03:30:00'], dtype="datetime64[ns, UTC]", freq=None),
         D=pd.DatetimeIndex(['2000-01-07 05:00:00', '2000-01-07 06:30:00'], dtype="datetime64[ns, UTC]", freq=None),
         E=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None),
     ))
])
def test_missing_indexes(data, expected):
    ds = DataSet(data)
    missing = ds.missing_indexes(sched, "1.5H")

    assert isinstance(missing, Data) and not isinstance(missing, DataSet)
    assert u.dict_same(missing.compute(), expected)


