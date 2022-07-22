from dask.delayed import delayed

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





