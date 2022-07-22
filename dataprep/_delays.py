from dask.delayed import delayed


@delayed
def _sessions(ix, tf):
    """
    How to group by sessions?
        find the start (diff > tf)
        allocate session markers
        group by them

    :param df:
    :return:
    """
    return ix.where(~(ix - ix.shift()).le(tf), None).ffill()

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
def missing_sessions(dfix, ix, tf):
    """
    reindex   (get none where missing)
    groupby sessions, count (to get the number of values in each session)

    select all those where count eq 0 from reindexed
    groupby session and select first value

    :param dfix:
    :param ix:
    :return:
    """
    redfix = dfix.to_series().reindex(ix)
    sessions = _sessions(redfix, tf)
    missing = redfix.groupby(sessions).transform("count").eq(0)
    return dfix[missing].groupby(sessions).first().index


def incomplete_sessions(self):
    sessions = self._reindexed_sessions(self.columns[0])

    count = sessions.transform("count")
    not_missing_but_incomplete = count.lt(sessions.transform("size")) & count.ne(0)

    df = self._reindexed[not_missing_but_incomplete]
    sessions = self._sessions(df)
    return df[sessions.cumcount().eq(0)].index.to_series().reset_index(drop=True)

