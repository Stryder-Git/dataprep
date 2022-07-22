from dask.delayed import delayed

@delayed
def missing(self):
    # drop all days that were missing entirely
    not_added_sessions = self._reindexed_sessions(self.columns[0]).transform("count").ne(0)
    df = self._reindexed[not_added_sessions]
    return df[df.isna().all(axis=1)]


def missing_indexes(self):
    nas = self._missing
    return nas.index.to_series().reset_index(drop=True)

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
    return counts[counts.eq(0)]

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
    return counts[counts.ne(size) & counts.ne(0)]





