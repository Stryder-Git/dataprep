import pytest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np
import pandas_market_calendars as mcal
import dataprep as dp
from dataprep import utils

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
    ds = dp.from_pandas(data)

    assert ds.all_columns_equal is False
    assert len(ds.common_columns) != ds.all_columns.str.len().max()

    ds = ds[ds.common_columns]
    assert ds.all_columns_equal


@pytest.mark.parametrize("data", test_data)
def test_index_properties(data):

    ds = dp.from_pandas(data)
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
    ds = dp.from_pandas(data)

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
    ds = dp.from_pandas(data)
    missing = ds.missing_sessions(sched, "1.5H")

    assert isinstance(missing, dp.Data) and not isinstance(missing, dp.DataSet)
    u.dict_same(missing.compute(), expected)


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
    ds = dp.from_pandas(data)
    incomplete = ds.incomplete_sessions(sched, "1.5H")

    assert isinstance(incomplete, dp.Data) and not isinstance(incomplete, dp.DataSet)
    u.dict_same(incomplete.compute(), expected)


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
    ds = dp.from_pandas(data)
    incomp_miss = ds.incomplete_or_missing_sessions(sched, "1.5H")

    assert isinstance(incomp_miss, dp.Data) and not isinstance(incomp_miss, dp.DataSet)
    u.dict_same(incomp_miss.compute(), expected)


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
    ds = dp.from_pandas(data)
    missing = ds.missing_indexes(sched, "1.5H")

    assert isinstance(missing, dp.Data) and not isinstance(missing, dp.DataSet)
    u.dict_same(missing.compute(), expected)


# for teting
def make(ix, n): return pd.Series(range(len(ix)), index=pd.DatetimeIndex(ix), name=n)

firstix = make(["2022-06-15", '2022-06-16', '2022-06-16 15:00:00', "2022-06-20"], "one").index
first = make(["2022-05-15", "2022-05-20", "2022-06-15", "2022-06-15 00:00:01", "2022-06-16 12:00:00", "2022-06-17",
              "2022-06-20"], "two")

secondix = make(["2000-01-15", "2000-03-15", "2000-04-15", "2000-09-15", "2000-10-15"], "one").index
second = make(["2000-02-20", "2000-02-25", "2000-03-15 00:00:01", "2000-04-01"], "two")

adapt_results = [
    (firstix, first, pd.Series([1, 3, 3, 5]), dict(norm=True, ffill=True)),  # 1+ nTrue fTrue
    (firstix, first, pd.Series([1, 3, np.nan, 5]), dict(norm=True, ffill=False)),  # 1+ nTrue fFalse

    (secondix, second, pd.Series([np.nan, 1, 3, 3, 3]), dict(norm=False, ffill=True)),  # 1+ nFalse fTrue
    (secondix, second, pd.Series([np.nan, 1, 3, np.nan, np.nan]), dict(norm=False, ffill=False)),  # 1+ nFalse fFalse

    (firstix, first, pd.Series([3, 4, 4, 6]), dict(off=0, norm=True, ffill=True)),  # 0 nTrue fTrue
    (firstix, first, pd.Series([3, 4, np.nan, 6]), dict(off=0, norm=True, ffill=False)),  # 0 nTrue fFalse

    (secondix, second, pd.Series([np.nan, 1, 3, 3, 3]), dict(off=0, norm=False, ffill=True)),  # 0 nFalse fTrue
    (secondix, second, pd.Series([np.nan, 1, 3, np.nan, np.nan]), dict(off=0, norm=False, ffill=False)) # 0 nFalse fFalse
]

"""
what are the possiblities regarding data index relationship

    whole:
        start - before/after
        end - before/after
            = 4

            before/before
            before/after
            after/before
            after/after

    each index:

        new info on index
        new info between current and previous index
        neither
            = 3

    data per index:
        between current and previous
            zero new piece of info (poi) 
            one poi
            2+ pois



what are kwarg permutations

    off - 0/ 1+
    norm   - True/False
    ffill  - True/False
        = 8

    0 nTrue fTrue
    0 nTrue fFalse
    0 nFalse fTrue
    0 nFalse fFalse
    1+ nTrue fTrue
    1+ nTrue fFalse
    1+ nFalse fTrue
    1+ nFalse fFalse



"""

@pytest.mark.parametrize("index, data, result, kw", adapt_results)
def test_adpat(index, data, result, kw):
    result.index = index
    calced = utils.adapt(index, data, **kw)

    equals = result.eq(calced.fillna(-1), fill_value=-1).all()
    if kw.get("fromix", False) is True:
        equals = equals.all()

    assert equals, str(result) + "\n" + str(calced)


"""

Two datasets with each two dataframes


    
"""

firstixset = dp.from_pandas({"A": firstix, "B": firstix})
firstset = dp.from_pandas({"A": first, "B": first*2,})
secondixset = dp.from_pandas({"A": secondix, "B": secondix})
secondset = dp.from_pandas({"A": second, "B": second*2})


match_data = [
    (firstset, firstixset, dp.from_pandas({
        "A": pd.Series([1, 3, 3, 5], index= firstix, dtype= float),
        "B": pd.Series([1, 3, 3, 5], index= firstix, dtype= float)*2
    }), dict(norm=True, ffill=True)),  # 1+ nTrue fTrue

    (firstset, firstixset, dp.from_pandas({
        "A": pd.Series([1, 3, np.nan, 5], index= firstix),
        "B": pd.Series([1, 3, np.nan, 5], index= firstix)*2
    }), dict(norm=True, ffill=False)),  # 1+ nTrue fFalse

    (secondset, secondixset, dp.from_pandas({
        "A": pd.Series([np.nan, 1, 3, 3, 3], index= secondix),
        "B": pd.Series([np.nan, 1, 3, 3, 3], index= secondix)*2
    }), dict(norm=False, ffill=True)),  # 1+ nFalse fTrue

    (secondset, secondixset, dp.from_pandas({
        "A": pd.Series([np.nan, 1, 3, np.nan, np.nan], index= secondix),
        "B": pd.Series([np.nan, 1, 3, np.nan, np.nan], index= secondix)*2
    }), dict(norm=False, ffill=False)),  # 1+ nFalse fFalse

    (firstset, firstixset, dp.from_pandas({
        "A": pd.Series([3, 4, 4, 6], index= firstix, dtype= float),
        "B": pd.Series([3, 4, 4, 6], index= firstix, dtype= float)*2
    }), dict(off=0, norm=True, ffill=True)),  # 0 nTrue fTrue

    (firstset, firstixset, dp.from_pandas({
        "A": pd.Series([3, 4, np.nan, 6], index= firstix),
        "B": pd.Series([3, 4, np.nan, 6], index= firstix)*2
    }), dict(off=0, norm=True, ffill=False)),  # 0 nTrue fFalse

    (secondset, secondixset, dp.from_pandas({
        "A": pd.Series([np.nan, 1, 3, 3, 3], index= secondix),
        "B": pd.Series([np.nan, 1, 3, 3, 3], index= secondix)*2
    }), dict(off=0, norm=False, ffill=True)),  # 0 nFalse fTrue

    (secondset, secondixset, dp.from_pandas({
        "A": pd.Series([np.nan, 1, 3, np.nan, np.nan], index= secondix),
        "B": pd.Series([np.nan, 1, 3, np.nan, np.nan], index= secondix)*2
    }), dict(off=0, norm=False, ffill=False))
    # 0 nFalse fFalse
]

@pytest.mark.parametrize("datas, ixset, expected, kwargs", match_data)
def test_match(datas, ixset, expected, kwargs):

    result = datas.match(ixset, **kwargs)
    result = result.compute()
    expected = expected.compute()

    assert result.shape == expected.shape
    expected.columns = result.columns

    print(type(result), "\n", result)
    print(type(expected), "\n", expected)
    assert_frame_equal(result, expected)


#
#
# """
# * fmp returns need to be replaced with test_data
#
# * test threaded and no threaded
# * test with and without ix
# * test different by and whats
#
# """
#
#
# #### Income Statement
# def _is_getter(api, sym, *args, **kwargs):
#     return pd.read_csv(f"financial_statements\\{sym}_income_statement.csv")
#
#
# def _bs_getter(api, sym, *args, **kwargs):
#     return pd.read_csv(f"financial_statements\\{sym}_balance_sheet_statement.csv")
#
#
# def _cs_getter(api, sym, *args, **kwargs):
#     return pd.read_csv(f"financial_statements\\{sym}_cash_flow_statement.csv")
#
#
# fmp.income_statement = _is_getter
# fmp.balance_sheet_statement = _bs_getter
# fmp.cash_flow_statement = _cs_getter

