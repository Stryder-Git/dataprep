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
    Assumes that all values of `data` are of the same type

    :param data: dict containing pandas object
    :param kwargs: passed to dd.from_pandas constructor
    :return:
    """
    if args is None: args = ()
    if kwargs is None: kwargs = {}
    d = data[list(data.keys())[0]]
    if not kwargs: kwargs["npartitions"] = 1

    if isinstance(d, pd.DataFrame):
        cls = DataSet
    elif isinstance(d, pd.Series):
        cls = DataSeries
    else:
        raise ValueError("Needs to be pandas object")

    return cls({s: dd.from_pandas(d, **kwargs) for s, d in data.items()})

def from_files(*files, name= None):
    """
    assumes that all files are of same type: .csv, else parquet

    :param files:
    :param name:
    :return:
    """
    if name is None: name = lambda p: Path(p).parts[-1].split(".")[0]

    syms = list(map(name, files))
    assert len(set(syms)) == len(files)
    reader = dd.read_csv if files[0].endswith(".csv") else dd.read_parquet
    return DataSet({s: reader(f, *args, **kwargs) for s, f in zip(syms, files)})


class _DataBase:
    client = dask

    def __init__(self, data):
        self.data = data

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
        shapes = self.client.compute({s: d.shape for s, d in self})
        if isinstance(shapes, tuple): shapes = shapes[0]
        return self._explode(shapes).astype(np.int64)

    @property
    def symbols(self):
        return pd.Index(self.data.keys())

    def apply(self, func, *args, pass_sym= False, **kwargs):
        func = delayed(func)
        if pass_sym: return {s: func(d, s, *args, **kwargs) for s, d in self}
        else: return {s: func(d, *args, **kwargs) for s, d in self}

    def persist(self):
        pers = self.client.persist(self.data)
        if isinstance(pers, tuple): pers = pers[0]
        return self.__class__(pers)

    def compute(self):
        comped = self.client.compute(self.data)
        if isinstance(comped, tuple): return pd.concat(comped[0])
        return pd.concat(comped.result())

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

    def drop_empties(self):
        shapes = self._shapes
        return self.__class__({s: self.get(s) for s, shape in shapes.items() if shape[0]})

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
        res = self.client.compute(self.data)
        if isinstance(res, tuple): return res[0]
        return res

    def persist(self):
        res = self.client.persist(self.data)
        if isinstance(res, tuple): return res[0]
        return res

    def __repr__(self):
        rp = {s: self.get(s) for s in self.symbols[:3]}
        return repr(rp)

class DataSeries(_DataBase):
    _axes = (0,)
    _frame = False

    def _all_equal(self, axis):
        assert axis in self._axes, "invalid axis argument"
        if self.shapes[axis].nunique() > 1: return False

        which = "index" if not axis else "columns"
        ix = getattr(self.get(self.symbols[0]), which)
        equals = ix == ix
        for s, d in self:
            _ix = getattr(d, which)
            equals &= _ix == ix
            ix = _ix

        equals = self.client.compute(equals.all())
        if isinstance(equals, tuple): return equals[0]
        return equals

    @cached_property
    def index_ranges(self):
        ranges = self.client.compute({s: (self.get(s).index.min(), self.get(s).index.max())
                                      for s in self.symbols})
        if isinstance(ranges, tuple): ranges = ranges[0]
        return self._explode(ranges)

    @property
    def all_indexes_equal(self):
        return self._all_equal(0)

    def match(self, data, same_syms= True, name= None, **kwargs):
        """
        Will match each dask.dataframe in self with the index of the
        dataframe in data that has the same symbol
        """
        if same_syms:
            syms = self.symbols.sort_values()
            other_syms = data.symbols.sort_values()
            assert len(other_syms) == len(syms) and (other_syms == syms).all()
        else:
            assert self.symbols.isin(data.symbols).all()

        @delayed
        def ad(ix, other): return u.adapt(ix, other, **kwargs)
        return self.__class__({s: ad(data.get(s).index, d) for s, d in self})

    def join(self, with_symbols= True):
        if with_symbols:
            if self._frame:
                data = [d.assign(symbol= s) for s, d in self]
            else:
                data = [d.to_frame().assign(symbol= s) for s, d in self]

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

        if isinstance(sample, tuple): sample = sample[0]
        sample = pd.concat(sample)

        if shuffle:
            ixs = np.arange(n)
            generator.shuffle(ixs)
            sample = sample.iloc[ixs]

        if with_sym: return sample
        else: return sample.droplevel(0, axis= 0)

    def _sessions(self, func, schedule, tf):
        ic = IndexCalculator(schedule, tf)
        ranges = self.index_ranges
        sessions = ic.sessions

        results = {}
        for s, d in self:
            rs = ranges.loc[s]
            results[s] = func(d.index, ic.timex(frm=rs[0], to=rs[1]), sessions)

        return Data(results)

    def missing_sessions(self, schedule, tf):
        return self._sessions(u.missing_sessions, schedule, tf)

    def incomplete_sessions(self, schedule, tf):
        return self._sessions(u.incomplete_sessions, schedule, tf)

    def incomplete_or_missing_sessions(self, schedule, tf):
        return self._sessions(u.incomplete_or_missing, schedule, tf)

    def missing_indexes(self, schedule, tf):
        return self._sessions(u.missing_indexes, schedule, tf)

    def __repr__(self):
        syms = self.symbols[:3]
        dfs = {}
        for sym in syms:
            dfs[sym] = self.get(sym).head(2)

        dfs = self.client.compute(dfs)
        if isinstance(dfs, tuple): dfs = dfs[0]
        return "\n" + repr(pd.concat(dfs))


class DataSet(DataSeries):
    _axes = (0, 1)
    _frame = True

    def __init__(self, data, same_cols= False):
        self.data = data
        if same_cols: self.validate()

    def validate(self):
        first = self.data[self.symbols[0]]
        cols = first.columns
        for sym, ddf in self:
            try:
                assert self.client.compute((cols == ddf.columns).all()), "Some columns don't match"
            except AttributeError as e:
                raise ValueError("Not all are same type") from e
            cols = ddf.columns

    @cached_property
    def common_columns(self):
        return reduce(lambda p,c: p[p.isin(c)],
                      (v.columns for v in self.data.values()))

    @property
    def all_columns(self):
        return pd.Series({k: v.columns.tolist() for k, v in self.data.items()})

    @property
    def all_columns_equal(self):
        return self._all_equal(1)

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
        results = self.client.compute({s: show(d) for s, d in self})
        if isinstance(results, tuple): results = results[0]
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

