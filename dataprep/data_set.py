import os
import dask
import dask.dataframe as dd
from dask.delayed import delayed
import pandas as pd
import numpy as np
from functools import cached_property, reduce
from pathlib import Path
from index_calculator import IndexCalculator

import dataprep.utils as u

default_rng = np.random.default_rng()
is_list_like = pd.api.types.is_list_like

def from_pandas(data, name= None, **kwargs):
    """
    Assumes that all values of `data` are of the same type.
    If that is a pd.Series or pd.Index, it will be converted to a DataFrame, using
    `name` for the name of the resulting column.

    :param data: dict containing pandas objects of the same type
    :param name: (only used when pandas objects are not DataFrames)
        a valid name for a DataFrame column
    :param kwargs: passed to dd.from_pandas constructor
    :return: DataSet
    """
    if kwargs is None: kwargs = {}
    d = data[list(data.keys())[0]]
    if not kwargs: kwargs["npartitions"] = 1

    if isinstance(d, pd.DataFrame):
        maker = lambda x: dd.from_pandas(x, **kwargs)
    elif isinstance(d, (pd.Series, pd.Index)):
        if name is None:
            n = "data"
            maker = lambda x: dd.from_pandas(
                x.to_frame(name=n if x.name is None else x.name), **kwargs)
        else:
            maker = lambda x: dd.from_pandas(x.to_frame(name=name), **kwargs)
    else:
        raise ValueError("Needs to be pandas object")

    return DataSet({s: maker(d) for s, d in data.items()})

def from_files(files, name, dtindex= None, **kwargs):
    """
    assumes that all files are of same type: .csv, else parquet

    :param files: iterable of filenames to retrieve, or str implying a folder
    :param name: function to extract name from filename, if `files` is a folder
        this will only be applied to the filenames. If `files` is an iterable of
        paths, it will be applied to the paths given by the user.
    :param dtindex: str indicating datetime column to be used as index, pass None
        to skip this
    :return:
    """
    assert callable(name), "`name` needs to be a callable"

    if isinstance(files, str):
        files = [files + "\\" + f for f in os.listdir(files)]
        n = name
        name = lambda p: n(Path(p).parts[-1])

    syms = list(map(name, files))
    assert len(set(syms)) == len(syms)
    reader = dd.read_csv if files[0].endswith(".csv") else dd.read_parquet
    ds = {s: reader(f, **kwargs) for s, f in zip(syms, files)}
    if dtindex is None: return DataSet(ds)

    return DataSet({s: d.set_index(dd.to_datetime(d.pop(dtindex)),
                                   drop= True) for s, d in ds.items()})


class _DataBase:
    client = dask

    def __init__(self, data):
        if issubclass(data.__class__, _DataBase):
            data = data.data
        self.data = data

    @property
    def symbols(self):
        return pd.Index(self.data.keys())

    def apply(self, func, *args, pass_sym= False, **kwargs):
        func = delayed(func)
        if pass_sym: return Data({s: func(d, s, *args, **kwargs) for s, d in self})
        else: return Data({s: func(d, *args, **kwargs) for s, d in self})

    def _pc(self, what=None, func= None):
        func = self.client.compute if func is None else func
        res = func(self.data if what is None else what)
        if isinstance(res, tuple): return res[0]
        return res

    def persist(self):
        pers = self._pc(func= self.client.persist)
        return self.__class__(pers)

    def compute(self):
        return pd.concat(self._pc())

    def get(self, sym, wrap= False):
        if wrap: return self.__class__({sym: self.data[sym]})
        return self.data[sym]

    def select(self, symbols):
        return self.__class__({s: self.get(s) for s in symbols})

    def copy(self, deep=False):
        if deep:
            new = {s: d.copy() for s, d in self}
        else:
            new = self.data.copy()
        return self.__class__(new)

    def drop(self, symbols):
        syms = self.symbols
        return self.select(syms[~syms.isin(symbols)])

    def __len__(self): return len(self.data)

    def __iter__(self): return iter(self.data.items())

    def __call__(self, func, *args, name= None, pass_sym= False, **kwargs):
        """
        Will apply a function to the data. That function needs to return a pandas object.
        If it is one-dimensional, it will be forced to be 2-dimensional. When forcing this
        and `name` is not passed, func.__name__ will be the name of the resulting column, unless
        the returned object from func already has a name that is not None. When `name` is passed and
        the returned object is one-dimensional, `name` will be used.

        :param func:
        :param args:
        :param name:
        :param pass_sym:
        :param kwargs:
        :return: DataSet
        """
        def _func(*args, **kwargs):
            res = func(*args, **kwargs)
            assert isinstance(res, (pd.DataFrame, pd.Series, pd.Index)), \
                "`func` needs to return a pandas object, otherwise use .apply"

            if res.ndim == 1:
                if not name is None: res.name = name
                elif res.name is None: res.name = func.__name__
                res = res.to_frame()
            return res

        return DataSet(self.apply(_func, *args, pass_sym= pass_sym, **kwargs))

    def __setitem__(self, key, value):
        raise TypeError(f"{self.__class__} does not support __setitem__")

    def __getitem__(self, item):
        raise TypeError(f"{self.__class__} does not support __getitem__")


class Data(_DataBase):
    def compute(self):
        return self._pc()

    def persist(self):
        return self._pc(func= self.client.persist)

    def to_frame(self, name= None):
        return self(lambda d: d.to_frame(name= name))

    def __repr__(self):
        rp = {s: self.get(s) for s in self.symbols[:3]}
        return repr(rp)

class DataSet(_DataBase):
    def __init__(self, data, same_cols= False):
        super().__init__(data)
        if same_cols:
            assert self.all_columns_equal

    def _explode(self, ranges):
        ranges = pd.Series(ranges).explode()
        assert ranges.index.value_counts().eq(2).all()
        return pd.DataFrame({0: ranges[::2], 1: ranges[1::2]})

    @property
    def shape(self):
        return len(self), len(self.common_columns)

    @property
    def fshape(self):
        return self.shapes[0].sum(), len(self.common_columns)

    @cached_property
    def shapes(self):
        shapes = self._pc({s: d.shape for s, d in self})
        return self._explode(shapes).astype(np.int64)

    def drop_empties(self):
        shapes = self.shapes
        return self.select(shapes.index[shapes[0].gt(0)])

    @cached_property
    def common_columns(self):
        comm = reduce(lambda p,c: p[p.isin(c)],
                      (v.columns for v in self.data.values()))
        return self._pc(comm)

    def _all_equal(self, axis):
        if self.shapes[axis].nunique() > 1: return False

        which = "index" if not axis else "columns"
        ix = getattr(self.get(self.symbols[0]), which)
        equals = ix == ix
        for s, d in self:
            _ix = getattr(d, which)
            equals &= _ix == ix
            ix = _ix

        return self._pc(equals.all())

    @property
    def all_columns(self):
        cols = self._pc({k: v.columns.tolist() for k, v in self.data.items()})
        return pd.Series(cols)

    @property
    def columns(self):
        assert self.all_columns_equal, "Not all columns are equal, " \
                                       "use the properties all_columns and common_columns to inspect this"
        return self.common_columns

    @property
    def all_columns_equal(self):
        return self._all_equal(1)

    @cached_property
    def index_ranges(self):
        ranges = self._pc({s: (self.get(s).index.min(), self.get(s).index.max())
                           for s in self.symbols})
        return self._explode(ranges)

    @property
    def all_indexes_equal(self):
        return self._all_equal(0)

    @cached_property
    def timezones(self):
        return pd.Series(self.apply(lambda d: d.index.tz).compute())

    @property
    def all_timezones_equal(self):
        return self.timezones.nunique(dropna= False) == 1

    @property
    def timezone(self):
        assert self.all_timezones_equal, "Not all timezones are equal"
        return self.timezones.iloc[0]

    @cached_property
    def frequencies(self):
        return pd.Series(self.apply(
            lambda d: (d.index[1:] - d.index[:-1]).value_counts().idxmax()
        ).compute())

    @property
    def all_frequencies_equal(self):
        return self.frequencies.nunique() == 1

    @property
    def frequency(self):
        assert self.all_frequencies_equal, "Not all frequencies are equal"
        return self.frequencies.iloc[0]


    def set_timezone(self, tz):
        """
        Converts timezone if dataframe is tz aware and localizes otherwise.
        :param tz:
        :return:
        """
        def settz(d):
            try: return d.tz_convert(tz)
            except TypeError: return d.tz_localize(tz)
        return self(settz)


    def match(self, data, off=0, day= pd.Timedelta("1D"), norm=False, ffill= False, fromix= False):
        """
        Will match each dataframe in self with the index of the dataframe in data that shares the symbol.
        The data will be matched to the equal or next larger index.

        IMPORTANT CAVEAT:

        This operation will *not* necessarily keep all data points. Therefore it is important
        to use absolute values. E.g.: use reported revenue and not yearly revenue growth.

            If there are two data points of the data between two indexes to be matched with or
             if there are duplicate indexes in the data, only the last data point will be kept:
             ```
                import dataprep as dp
                import pandas as pd

                data = pd.Series({"2000-01-05": 1, "2000-03-05": 2})
                data.index = pd.to_datetime(data.index)
                ix = pd.DatetimeIndex(["2000-01-01", "2000-06-01"])

                data = dp.from_pandas(dict(A= data))
                ix = dp.from_pandas(dict(A= ix))

                data.match(ix).compute()
                >>
                              data
                A 2000-01-01   NaN
                  2000-06-01   2.0
             ```

        Parameters
        ----------
        off : int, default 0
            how many times to add `day` to the data's index before matching
        day: pd.Timedelta, DateOffset, or BusinessDay, default: pd.Timedelta("1D")
           This will be multiplied with `off` and added to the DatetimeIndex of the data
        norm : bool, default False
            Normalize data's index to midnight before adapting.
        ffill : bool, default True
            ffill data, otherwise keep only values that are new.
        fromix : bool, default False
            return as DataFrame containing a "fromix" column holding the indexes of adapt
            that the values are from, ffill also applies to this.
            This column will contain the original dates (before norm/off was applied).

        Returns
        -------
        dp.DataSet

        """
        assert self.symbols.isin(data.symbols).all()
        data = data.data
        return DataSet({s: u.adapt(d, data[s].index, off, day,
                                   norm, ffill, fromix) for s, d in self})

    def join(self, with_symbols= True):
        if with_symbols:
            data = [d.assign(symbol= s) for s, d in self]

        else: data = list(self.data.values())
        try: full = dd.from_delayed(data)
        except TypeError: full = dd.concat(data)
        return full

    def ffill(self):
        return self.__class__(self.apply(lambda d: d.ffill()))

    def head(self, n= 5, i= 0):
        hd = self.data[self.symbols[i]].head(n)
        if isinstance(hd, (pd.DataFrame, pd.Series)): return hd
        return hd.compute()

    def sample(self, n= 1000, with_sym= False, shuffle= True, generator= default_rng):
        rows = self.shapes[0]
        cmsum = np.cumsum(rows)
        nrows = cmsum[-1]
        n = min(n, nrows)

        chose = np.sort(generator.choice(nrows, size= (n,), replace= False))
        split = np.split(chose, np.searchsorted(chose, cmsum))

        sample = dict()
        for sym, r, ixs in zip(rows.keys(), rows, split):
            sample[sym] = self.data[sym].iloc[ixs]
            ixs -= r

        sample = pd.concat(self._pc(sample))

        if shuffle:
            ixs = np.arange(n)
            generator.shuffle(ixs)
            sample = sample.iloc[ixs]

        if with_sym: return sample
        else: return sample.droplevel(0, axis= 0)

    def _sessions(self, func, schedule, freq):
        if pd.isna(self.timezone):
            assert schedule.apply(lambda c: c.dt.tz).isna().all(), "Cannot use a tz aware schedule on tz naive data"

        if freq is None: freq = self.frequency
        name = func.__name__
        ic = IndexCalculator(schedule, freq)
        ranges = self.index_ranges
        sessions = ic.sessions

        results = {}
        for s, d in self:
            rs = ranges.loc[s]
            results[s] = func(d.index, ic.timex(frm=rs[0], to=rs[1]), sessions
                              ).to_frame(name= name)

        return DataSet(results)

    def missing_sessions(self, schedule, freq= None):
        return self._sessions(u.missing_sessions, schedule, freq)

    def incomplete_sessions(self, schedule, freq= None):
        return self._sessions(u.incomplete_sessions, schedule, freq)

    def incomplete_or_missing_sessions(self, schedule, freq= None):
        return self._sessions(u.incomplete_or_missing_sessions, schedule, freq)

    def missing_indexes(self, schedule, freq= None):
        return self._sessions(u.missing_indexes, schedule, freq)

    def ffill_sessions(self, schedule, freq= None):
        """
        naive ffill of all sessions followed by
         bfill in case the start of the first session is missing
        """
        if pd.isna(self.timezone):
            assert schedule.apply(lambda c: c.dt.tz).isna().all(), "Cannot use a tz aware schedule on tz naive data"

        if freq is None: freq = self.frequency
        ic = IndexCalculator(schedule, freq)
        ranges = self.index_ranges

        results = {}
        for s, d in self:
            rs = ranges.loc[s]
            results[s] = u.ffill_sessions(d, ic.timex(frm=rs[0], to=rs[1]))

        return DataSet(results)

    def __repr__(self):
        syms = self.symbols[:3]
        dfs = {}
        for sym in syms:
            dfs[sym] = self.get(sym).head(2)

        return "\n" + repr(pd.concat(self._pc(dfs)))

    def show_issues(self, absolute= True):
        """
        what do I want
            for each symbol
                for each column
                    nan
                    inf
        :return:
        """

        def issues(d):
            numeric = d.select_dtypes(exclude= ["string", "object"])
            return pd.concat(dict(inf= numeric.abs().eq(np.inf).sum(),
                                  nan= d.isna().sum()), axis= 1)
        if absolute: show = issues
        else:
            def show(d): return issues(d)/d.shape[0]

        show = delayed(show)
        results = self._pc({s: show(d) for s, d in self})
        return pd.concat(results).unstack(-1)

    def __getitem__(self, item):
        if not is_list_like(item): item = [item]
        item = list(item)
        assert pd.Index(item).isin(self.common_columns).all()
        return self(lambda d: d[item])

    def __setitem__(self, key, value):
        if not isinstance(value, DataSet):
            raise NotImplementedError("So far only possible with DataSet")
        else:
            assert len(self) == len(value)
            assert (self.symbols.sort_values() == value.symbols.sort_values()).all()

        if not is_list_like(key): key = [key]

        @delayed
        def _set(d, k, v):
            d[k] = v
            return d

        for s in self.symbols:
            other = value.get(s)
            self.data[s] = _set(self.data[s], key, other[key])

