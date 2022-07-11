import pytest
import pandas as pd
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
def test_all_columns_equal(data):
    ds = DataSet(data)
    assert ds.all_columns_equal is False

    ds = ds["a", "b", "d"]
    assert ds.all_columns_equal


@pytest.mark.parametrize("data", test_data)
def test_all_indexes_equal(data):

    ds = DataSet(data)
    assert ds.all_indexes_equal is False

    ds = ds.select(list("AB"))
    assert ds.all_indexes_equal





