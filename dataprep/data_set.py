


class DataSet:

    client = dask

    def validate(self):
        first = self.data[self.symbols[0]]
        cols = first.columns
        for sym, ddf in self:
            try:
                assert self.client.compute((cols == ddf.columns).all()), "Some columns don't match"
            except AttributeError as e:
                raise ValueError("Not all are same type") from e
            cols = ddf.columns

    @classmethod
    def __new(cls, *files, name= None, args= None, kwargs= None, self= None):
        self = super().__new__(cls) if self is None else self
        if args is None: args = ()
        if kwargs is None: kwargs = {}

        if isinstance(files[0], dict):
            self.data = files[0].copy()
            d = self.get(self.symbols[0])
            if isinstance(d, (pd.DataFrame, pd.Series)):
                if not kwargs: kwargs["npartitions"] = 1
                if d.ndim == 1:
                    d = {s: dd.from_pandas(d.to_frame(name), **kwargs) for s, d in self}
                else:
                    d = {s: dd.from_pandas(d, **kwargs) for s, d in self}
                self.data = d

        elif isinstance(files[0], pd.DataFrame):
            df = files[0]
            assert df.columns.nlevels <= 2 and df.index.nlevels == 1
            if not kwargs: kwargs["npartitions"] = 1

            data = {}
            for sym in df.columns.get_level_values(0):
                vals = df[sym]
                if vals.ndim == 1:
                    assert not name is None
                    vals = vals.to_frame(name)
                data[sym] = dd.from_pandas(vals, *args, **kwargs)

            self.data = data

        else:
            if name is None: name = lambda p: Path(p).parts[-1].split(".")[0]

            syms = list(map(name, files))
            assert len(set(syms)) == len(files)
            reader = dd.read_csv if files[0].endswith(".csv") else dd.read_parquet
            self.data = {s: reader(f, *args, **kwargs) for s, f in zip(syms, files)}

        return self

    def __init__(self, *files, name=None, args=None, kwargs=None, verify_integrity= False):
        self.__new(*files, name=name, args=args, kwargs=kwargs, self= self)
        if verify_integrity:
            self.validate()

    @cached_property
    def nrows(self):
        shapes = self.client.compute((x.shape[0] for _, x in self))
        if isinstance(shapes, tuple): shapes = shapes[0]
        return sum(shapes)

    @property
    def ncols(self):
        return self.columns.shape[0]

    @property
    def shape(self):
        return len(self), self.ncols

    @property
    def fshape(self):
        return self.nrows, self.ncols

    @cached_property
    def _shapes(self):
        shapes = self.client.compute({s: d.shape for s, d in self})
        if isinstance(shapes, tuple): shapes = shapes[0]
        return shapes

    def _explode(self, ranges):
        ranges = pd.Series(ranges).explode()
        assert ranges.index.value_counts().eq(2).all()
        return pd.DataFrame({0: ranges[::2], 1: ranges[1::2]})

    @property
    def shapes(self):
        return self._explode(self._shapes)

    @cached_property
    def index_ranges(self):
        ranges = self.client.compute({s: (self.get(s).index.min(), self.get(s).index.max())
                                      for s in self.symbols})
        if isinstance(ranges, tuple): ranges = ranges[0]
        return self._explode(ranges)

    @property
    def symbols(self):
        return pd.Index(self.data.keys())

    @property
    def columns(self):
        first = self.data[self.symbols[0]]
        cols = first.columns
        cols = self.client.compute(cols)
        if isinstance(cols, tuple): cols = cols[0]
        return cols

    def _all_equal(self, axis):
        assert axis in (0, 1), "invalid axis argument"
        if not len(set(s[axis] for s in self._shapes.values())) == 1:
            return False

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

    @property
    def all_columns_equal(self):
        return self._all_equal(1)

    @property
    def all_indexes_equal(self):
        return self._all_equal(0)

    def apply(self, func, *args, pass_sym= False, **kwargs):
        func = delayed(func)
        if pass_sym: return {s: func(d, s, *args, **kwargs) for s, d in self}
        else: return {s: func(d, *args, **kwargs) for s, d in self}

    def persist(self):
        pers = self.client.persist(self.data)
        if isinstance(pers, tuple): pers = pers[0]
        return self.__new(pers)

    def compute(self):
        comped = self.client.compute(self.data)
        if isinstance(comped, tuple): return pd.concat(comped[0])
        return pd.concat(comped.result())

    def get(self, sym, wrap= False):
        if wrap: return self.__new({sym: self.data[sym]})
        return self.data[sym]

    def __getitem__(self, item):
        if not pd.api.types.is_list_like(item): item = [item]
        item = list(item)
        assert pd.Index(item).isin(self.columns).all()
        return self(lambda d: d[item])

    def __setitem__(self, key, value):
        if not isinstance(value, DataSet):
            raise NotImplementedError("So far only possible with DataSet")
        else:
            assert len(self) == len(value)
            assert (self.symbols.sort_values() == value.symbols.sort_values()).all()

        if not isinstance(key, (list, tuple, pd.Index)): key = [key]

        @delayed
        def _set(d, k, v):
            d[k] = v
            return d

        for s in self.symbols:
            other = value.get(s)
            self.data[s] = _set(self.data[s], key, other[key])

    def head(self, n= 5, i= 0):
        hd = self.data[self.symbols[i]].head(n)
        if isinstance(hd, pd.DataFrame): return hd
        return hd.compute()

    def sample(self, n= 1000, with_sym= False):
        n = min(n, self.nrows)
        ks = self.data.keys()
        rows = [self.get(s, wrap=True).fshape[0] for s in ks]
        cmsum = np.cumsum(rows)

        chose = npr.choice(np.sum(rows), size= (n,), replace= False)
        chose.sort()
        split = np.split(chose, np.searchsorted(chose, cmsum))

        taken = 0
        sample = dict()
        for sym, r, ixs in zip(ks, rows, split):
            sample[sym] = self.data[sym].iloc[ixs - taken]
            taken += r

        sample = self.client.compute(sample)
        if isinstance(sample, tuple): sample = sample[0]

        sample = pd.concat(sample)
        ixs = np.arange(n)
        npr.shuffle(ixs)
        sample = sample.iloc[ixs]
        if with_sym: return sample
        else: return sample.droplevel(0, axis= 0)

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
        def _adj(d, other):
            adapted = adapt(d.index, other)
            if adapted.ndim == 1: return adapted.to_frame(name)
            return adapted

        return self.__new({s: _adj(data.get(s), d) for s, d in self})

    def join(self, with_symbols= True):
        if with_symbols: data = [d.assign(symbol= s) for s, d in self]
        else: data = list(self.data.values())
        try: full = dd.from_delayed(data)
        except TypeError: full = dd.concat(data)
        return full
        # reset_index resets it per partition, not what I thought
        # if full.index.name in full.columns:
        #     warnings.warn("Not resetting index because a column with its name already exists")
        #     return full
        # return full.reset_index(drop= False)

    def ffill(self): return self(lambda d: d.ffill())

    def __len__(self): return len(self.data)

    def __iter__(self): return iter(self.data.items())

    def __call__(self, func, *args, n= None, pass_sym= False, **kwargs):
        def _func(*args, **kwargs):
            res = func(*args, **kwargs)
            if res.ndim == 1:
                if not n is None: res.name = n
                elif res.name is None: res.name = func.__name__
                res = res.to_frame()
            return res

        return self.__new(self.apply(_func, *args, pass_sym= pass_sym, **kwargs))

    def copy(self, deep= False):
        if deep: new = {s: d.copy() for s, d in self}
        else: new = self.data.copy()
        return self.__new(new)

    def __repr__(self):
        syms = self.symbols[:3]
        dfs = {}
        for sym in syms:
            dfs[sym] = self.get(sym).head(2)

        dfs = self.client.compute(dfs)
        if isinstance(dfs, tuple): dfs = dfs[0]
        return "\n" + repr(pd.concat(dfs))

    def drop_empties(self):
        shapes = self._shapes
        return self.__new({s: self.get(s) for s, shape in shapes.items() if shape[0]})

    def select(self, symbols):
        return self.__new({s: self.get(s) for s in symbols})

    def drop(self, symbols):
        syms = self.symbols
        return self.select(syms[~syms.isin(symbols)])

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

        if absolute:
            show = issues
        else:
            def show(d): return issues(d)/d.shape[0]

        show = delayed(show)
        results = self.client.compute({s: show(d) for s, d in self})
        if isinstance(results, tuple): results = results[0]
        return pd.concat(results).unstack(-1)

    def describe(self):
        results = self.client.compute({s: d.describe() for s, d in self})






    """
    missing_sessions
        uses reindexed index column
        groupby sessions
        uses .transform(count).eq(0) to determine what ixs are missing   
        
        then from the reindexed df, 
            select missing
        missing.groupby sessions,
            cumcount().eq(0) (to get the first row of the df
        
    incomplete_sessions
        
        
        
        
    
    missing_indexes
    """


