import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from dataprep import DataSet


"""
Permutations

    columns that are not in all

    indexes covering much less data

    indexes that have missing rows

    indexes that have missing sessions

        include breaks


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


def test_missing_sessions(data):
    ds = DataSet(data)






