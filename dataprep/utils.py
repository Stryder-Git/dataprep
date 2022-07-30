from dask.delayed import delayed
import pandas as pd
import numpy as np

from pandas_market_calendars import get_calendar
from pandas.tseries.offsets import CustomBusinessDay
import warnings

# _nyse = get_calendar("NYSE")
# nyse_busday = CustomBusinessDay(weekmask= "1111100",
#                            holidays= _nyse.adhoc_holidays,
#                            calendar= _nyse.regular_holidays)

@delayed
def adapt(data, ix, off, day, norm, ffill, fromix):

    assert ix.is_monotonic_increasing and not ix.duplicated().any()
    ndim = data.ndim
    if ndim == 1: data = data.to_frame()
    orig_index = data.index.copy()
    data = data.assign(fromix= orig_index)
    data = data.dropna(how="any")

    if norm: data.index = data.index.normalize()
    if off: data.index += off * day

    duplicated = data.index.duplicated(keep="last")
    warn = duplicated.any()
    if warn: data = data[~data.index.duplicated(keep="last")]

    data = data.reindex(ix.union(data.index)).ffill()
    if ffill: data = data.loc[ix]
    else:
        data = data.loc[ix]
        notnew = data.fromix.eq(data.fromix.shift(1))
        data.loc[notnew] = np.nan

    if warn or not orig_index.isin(data.fromix).all():
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

