from dask.delayed import delayed
import pandas as pd
import numpy as np

from pandas_market_calendars import get_calendar
from pandas.tseries.offsets import CustomBusinessDay
import warnings

_nyse = get_calendar("NYSE")
nyse_busday = CustomBusinessDay(weekmask= "1111100",
                           holidays= _nyse.adhoc_holidays,
                           calendar= _nyse.regular_holidays)

one_day = pd.Timedelta("1D")

def adapt(ix, data, off=1, day= one_day, norm=True, ffill= False, fromix= False):
    """
    This function maps each data point of `adapt` to the equal or next larger index
    in `ix`, after normalizing and/or adding `off` * `day` to adapt.index.

    CAVEAT:

    This operation will *not* necessarily keep all data points from `adapt`. Therefore it is important
    to use absolute values. E.g.: use reported revenue and not yearly revenue growth.
    Although, a warning will be raised if something is lost.

        If there are two data points of `adapt` between two indexes of `ix`, only the last data point
        will be kept:
            import research_environment as renv
            import pandas as pd

            ix = pd.DatetimeIndex(["2000-01-01", "2000-06-01"])
            data = pd.Series({"2000-01-05": 1, "2000-03-05": 2})
            data.index = pd.to_datetime(data.index)

            renv.utils.adapt(ix, data)
            >>
             2000-01-01 NaN
             2000-06-01 2

    Parameters
    ----------
    off : int, default 1
        how many times to add `day` to adapt.index before adapting
    norm : bool, default True
        Normalize index to midnight before adapting.
    ffill : bool, default True
        fill all values for `ix`, else keep only values that are new.
    fromix : bool, default False
        return as DataFrame containing a "fromix" column holding the indexes of adapt
        that the values are from, ffill also applies to this.
        This column will contain the original dates (before norm/off was applied).

    Returns
    -------
    pandas.Series if not fromix (default) else pandas.DataFrame

    """
    assert ix.is_monotonic_increasing and not ix.duplicated().any()
    ndim = data.ndim
    if ndim == 1: data = data.to_frame()
    orig_index = data.index.copy()
    data = data.assign(fromix= orig_index)
    data = data.dropna(how="any")

    if norm: data.index = data.index.normalize()
    if off: data.index += off * day

    duplicated = data.index.duplicated(keep="last")
    if duplicated.any():
        data = data[~data.index.duplicated(keep="last")]

    data = data.reindex(ix.union(data.index)).ffill()
    if ffill: data = data.loc[ix]
    else:
        data = data.loc[ix]
        notnew = data.fromix.eq(data.fromix.shift(1))
        data.loc[notnew] = np.nan

    if not orig_index.isin(data.fromix).all():
        warnings.warn("Some values are lost")

    if fromix: return data
    elif ndim == 1: return data.iloc[:, 0]
    return data.drop(columns= ["fromix"])



@delayed
def missing_sessions(dfix, ix, sessions):
    """
    get


    :param dfix:
    :param ix:
    :return:
    """
    redfix = dfix.to_series().reindex(ix)
    grper = ix.to_series().where(ix.isin(sessions)).ffill()
    counts = redfix.groupby(grper).count()
    return counts.index[counts.eq(0)]

@delayed
def incomplete_sessions(dfix, ix, sessions):
    """
    get counts of

    :param self:
    :return:
    """
    redfix = dfix.to_series().reindex(ix)
    grper = ix.to_series().where(ix.isin(sessions)).ffill()
    grp = redfix.groupby(grper)
    counts = grp.count()
    size = grp.size()
    return counts.index[counts.ne(size) & counts.ne(0)]

@delayed
def incomplete_or_missing_sessions(dfix, ix, sessions):
    redfix = dfix.to_series().reindex(ix)
    grper = ix.to_series().where(ix.isin(sessions)).ffill()
    grp = redfix.groupby(grper)
    counts = grp.count()
    size = grp.size()
    return counts.index[counts.ne(size)]

@delayed
def missing_indexes(dfix, ix, sessions):  # keeping `sessions` so the signature is the same as above
    redfix = dfix.to_series().reindex(ix)
    return redfix.index[redfix.isna()]

@delayed
def ffill_sessions(df, ix):
    return df.reindex(ix).ffill().bfill()


def reset_pck(pck):
    from sys import modules
    for k in modules.copy():
        if pck in k: del modules[k]

