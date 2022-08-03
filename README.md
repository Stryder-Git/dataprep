## This package provides a three dimensional data structure, building on dask dataframes and delayed objects.
The main class is `DataSet` and it should be used for embarrassingly parallel computations on a large number of dataframes. Rather than a full replacement for the dask api, this is intended to be an extension, with particular focus on financial timeseries data.

```python
# The following code is based on examples\examples.ipynb

import os
import pandas as pd
import dataprep as dp
import pandas_market_calendars as mcal
```

## Read Data and Basic Properties


```python
data = dp.from_files("sample_data", name= lambda p: p.split("_")[0], dtindex= "date")

data = data.set_timezone("America/New_York")
```


```python
data.symbols
```




    Index(['AAPL', 'AMZN', 'NVDA'], dtype='object')




```python
data.frequency
```




    Timedelta('0 days 00:30:00')




```python
data.timezone
```




    <DstTzInfo 'America/New_York' LMT-1 day, 19:04:00 STD>




```python
data.symbols
```




    Index(['AAPL', 'AMZN', 'NVDA'], dtype='object')




```python
data.columns
```




    Index(['open', 'high', 'low', 'close', 'volume'], dtype='object')




```python
data.all_indexes_equal
```




    False




```python
data.index_ranges # they all cover the same date ranges
```




                                 0                         1
    AAPL 2020-01-02 04:00:00-05:00 2020-01-15 19:30:00-05:00
    AMZN 2020-01-02 04:00:00-05:00 2020-01-15 19:30:00-05:00
    NVDA 2020-01-02 04:00:00-05:00 2020-01-15 19:30:00-05:00




```python
data.shapes # shape of each dataframe ... something is off about AMZN
```




            0  1
    AAPL  320  5
    AMZN  312  5
    NVDA  320  5




```python
data.shape, data.fshape  # shape of the DataSet (symbols, columns), full shape of data (sum of rows, columns)
```




    ((3, 5), (952, 5))



## Inspecting Price Data for Incomplete and Missing Sessions

The following methods need a schedule dataframe, as it is provided by the packages `pandas_market_calendars` and `exchange_calendars`.


```python
nyse = mcal.get_calendar("NYSE").schedule("2020-01-01", "2020-01-20", market_times= "all", tz= data.timezone)
```


```python
missing = data.missing_sessions(nyse)

missing.compute() # There are no sessions that are missing entirely...
```




    Empty DataFrame
    Columns: [missing_sessions]
    Index: []




```python
incomp = data.incomplete_sessions(nyse)

incomp.compute() # But AMZN has three incomplete sessions
```




                                         incomplete_sessions
    AMZN 2020-01-09 04:00:00-05:00 2020-01-09 04:00:00-05:00
         2020-01-10 04:00:00-05:00 2020-01-10 04:00:00-05:00
         2020-01-15 04:00:00-05:00 2020-01-15 04:00:00-05:00




```python
missing = data.missing_indexes(nyse)

# These are the indexes that should exist according to NYSE's schedule, 
missing.compute()
```




                                             missing_indexes
    AMZN 2020-01-09 04:00:00-05:00 2020-01-09 04:00:00-05:00
         2020-01-09 04:30:00-05:00 2020-01-09 04:30:00-05:00
         2020-01-10 04:00:00-05:00 2020-01-10 04:00:00-05:00
         2020-01-10 04:30:00-05:00 2020-01-10 04:30:00-05:00
         2020-01-10 05:00:00-05:00 2020-01-10 05:00:00-05:00
         2020-01-10 05:30:00-05:00 2020-01-10 05:30:00-05:00
         2020-01-10 06:00:00-05:00 2020-01-10 06:00:00-05:00
         2020-01-15 04:00:00-05:00 2020-01-15 04:00:00-05:00




```python
# This explains the difference between the shapes
pd.concat([data.shapes[0], missing.shapes[0]], axis= 1)
```




            0  0
    AAPL  320  0
    AMZN  312  8
    NVDA  320  0



### Fill the data to have complete indexes

`DataSet.ffill_sessions` will use `pd.DataFrame.ffill` in a way that all indexes
that should exist according to NYSE's schedule are represented with the previous index's value.


```python
filled = data.ffill_sessions(nyse) 
print(filled.all_indexes_equal)
filled.shapes
```

    True
    
            0  1
    AAPL  320  5
    AMZN  320  5
    NVDA  320  5



## Apply Any Python Function

There are two options:

    * If the function returns a pandas object, call the DataSet directly.
    * If not, use .apply.


```python
# This returns a pd.Series
def av_pre_volume(d):
    d = d[(d.index - d.index.normalize()) < pd.Timedelta("9.5H")]
    return d["volume"].groupby(d.index.normalize()).mean()
```


```python
av_pre_vol = filled(av_pre_volume)
```

`DataSet.__call__` will return a `DataSet` object. If the return value of the passed function is one dimensional, it will be converted to a `DataFrame` to make it fit into a `DataSet`. So far, there is no specific support for `pd.Index` or `pd.Series`.


```python
av_pre_vol
```




    
                                         volume
    AAPL 2020-01-02 00:00:00-05:00  1654.090909
         2020-01-03 00:00:00-05:00  3615.454545
    AMZN 2020-01-02 00:00:00-05:00   565.727273
         2020-01-03 00:00:00-05:00  1043.545455
    NVDA 2020-01-02 00:00:00-05:00   283.909091
         2020-01-03 00:00:00-05:00   359.363636




```python
# This returns a scalar
def av_pre_volume(d):
    pre = (d.index - d.index.normalize()) < pd.Timedelta("9.5H")
    return d.loc[pre, "volume"].mean()
```


```python
av_pre_vol = filled.apply(av_pre_volume)
```

`DataSet.apply` will return a `Data` object, which doesn't have most of the features that a `DataSet` has because 
it doesn't assume that it will deal with pandas objects.


```python
av_pre_vol.compute()
```




    {'AAPL': 2130.2363636363634, 'AMZN': 474.4818181818182, 'NVDA': 259.4}



If you pass this to the DataSet directly it will raise an error when the delayed object is computed.


```python
try:
    filled(av_pre_volume).compute()
except AssertionError as e:
    print(e)
```

    `func` needs to return a pandas object, otherwise use .apply
    

## Match Data with Different Frequencies


```python
# For demonstration, convert the data to a daily frequency
def todaily(d):
    return d.resample("1D", origin= "start").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

daily = filled(todaily)

daily.frequency
```




    Timedelta('1 days 00:00:00')




```python
daily.compute()
```




                                     open   high    low  close     volume
    AAPL 2020-01-02 04:00:00-05:00  73.76  75.30  73.76  75.30  1168200.0
         2020-01-03 04:00:00-05:00  74.34  75.14  73.60  74.33  1271678.0
         2020-01-06 04:00:00-05:00  73.86  74.99  73.19  74.91  1007688.0
         2020-01-07 04:00:00-05:00  75.12  75.23  73.00  73.15   916430.0
         2020-01-08 04:00:00-05:00  74.13  76.11  74.13  76.00  1128600.0
         2020-01-09 04:00:00-05:00  76.30  77.61  76.26  77.60  1483274.0
         2020-01-10 04:00:00-05:00  77.81  78.17  77.06  77.67  1221207.0
         2020-01-13 04:00:00-05:00  77.88  79.70  77.79  79.70  1051003.0
         2020-01-14 04:00:00-05:00  79.13  79.44  77.70  77.94  1394565.0
         2020-01-15 04:00:00-05:00  77.97  78.88  77.39  78.05  1032766.0
    AMZN 2020-01-02 04:00:00-05:00  93.21  95.02  92.39  94.99   487803.0
         2020-01-03 04:00:00-05:00  93.50  94.31  93.00  93.57   405417.0
         2020-01-06 04:00:00-05:00  92.99  95.18  92.67  95.08   481008.0
         2020-01-07 04:00:00-05:00  95.46  95.69  92.75  93.15   470907.0
         2020-01-08 04:00:00-05:00  94.80  95.58  94.32  94.78   400591.0
         2020-01-09 04:00:00-05:00  94.78  95.89  94.78  95.17   378158.0
         2020-01-10 04:00:00-05:00  95.09  95.40  94.00  94.06   336750.0
         2020-01-13 04:00:00-05:00  94.48  94.90  94.04  94.63   324663.0
         2020-01-14 04:00:00-05:00  94.25  94.61  92.93  93.25   384719.0
         2020-01-15 04:00:00-05:00  93.25  93.94  92.75  93.24   317252.0
    NVDA 2020-01-02 04:00:00-05:00  59.26  60.10  59.18  60.01   180489.0
         2020-01-03 04:00:00-05:00  59.13  59.46  58.04  59.02   157305.0
         2020-01-06 04:00:00-05:00  58.25  59.35  57.69  59.35   209028.0
         2020-01-07 04:00:00-05:00  59.55  60.44  57.63  57.88   258745.0
         2020-01-08 04:00:00-05:00  59.33  60.51  59.33  60.25   224926.0
         2020-01-09 04:00:00-05:00  60.63  61.48  60.21  60.93   203703.0
         2020-01-10 04:00:00-05:00  61.22  62.14  60.94  61.08   253594.0
         2020-01-13 04:00:00-05:00  61.72  63.25  61.52  63.20   257304.0
         2020-01-14 04:00:00-05:00  62.93  63.22  61.67  61.68   286669.0
         2020-01-15 04:00:00-05:00  61.88  62.75  61.13  61.54   203550.0




```python
# Create demo data
data = dict(revcost= range(4), revenue= range(4, 8))
ix = lambda x: pd.DatetimeIndex(x).tz_localize(daily.timezone)
funds = dict(
    AAPL = pd.DataFrame(data, index= ix(["2020-01-03", "2020-01-07", "2020-01-11", "2020-01-14"])),
    NVDA = pd.DataFrame(data, index= ix(["2020-01-01", "2020-01-05", "2020-01-05", "2020-01-12"])),
    AMZN = pd.DataFrame(data, index= ix(["2020-01-06", "2020-01-08 14:30:00", "2020-01-09", "2020-01-13"]))
)
```


```python
funds = dp.from_pandas(funds)

funds.compute()
```




                                    revcost  revenue
    AAPL 2020-01-03 00:00:00-05:00        0        4
         2020-01-07 00:00:00-05:00        1        5
         2020-01-11 00:00:00-05:00        2        6
         2020-01-14 00:00:00-05:00        3        7
    NVDA 2020-01-01 00:00:00-05:00        0        4
         2020-01-05 00:00:00-05:00        1        5
         2020-01-05 00:00:00-05:00        2        6
         2020-01-12 00:00:00-05:00        3        7
    AMZN 2020-01-06 00:00:00-05:00        0        4
         2020-01-08 14:30:00-05:00        1        5
         2020-01-09 00:00:00-05:00        2        6
         2020-01-13 00:00:00-05:00        3        7



The `DataSet.match` method expects a `DataSet` as first argument and will match the data of the caller to the indexes of the passed DataSet, which requires matching symbols. There are multiple keyword arguments that allow you to control the way the data is matched.


```python
# fromix = True will add the index that the data is from as a column
matched = funds.match(daily, fromix= True)

matched.compute().dropna(how= "all")
```




                                    revcost  revenue                    fromix
    AAPL 2020-01-03 04:00:00-05:00      0.0      4.0 2020-01-03 00:00:00-05:00
         2020-01-07 04:00:00-05:00      1.0      5.0 2020-01-07 00:00:00-05:00
         2020-01-13 04:00:00-05:00      2.0      6.0 2020-01-11 00:00:00-05:00
         2020-01-14 04:00:00-05:00      3.0      7.0 2020-01-14 00:00:00-05:00
    NVDA 2020-01-02 04:00:00-05:00      0.0      4.0 2020-01-01 00:00:00-05:00
         2020-01-06 04:00:00-05:00      2.0      6.0 2020-01-05 00:00:00-05:00
         2020-01-13 04:00:00-05:00      3.0      7.0 2020-01-12 00:00:00-05:00
    AMZN 2020-01-06 04:00:00-05:00      0.0      4.0 2020-01-06 00:00:00-05:00
         2020-01-09 04:00:00-05:00      2.0      6.0 2020-01-09 00:00:00-05:00
         2020-01-13 04:00:00-05:00      3.0      7.0 2020-01-13 00:00:00-05:00




```python
# off= 1 will add one day to the data's index before matching, norm= True will also normalize it.
matched = funds.match(daily, off= 1, norm= True, fromix= True)
matched.compute().dropna(how= "all")
```




                                    revcost  revenue                    fromix
    AAPL 2020-01-06 04:00:00-05:00      0.0      4.0 2020-01-03 00:00:00-05:00
         2020-01-08 04:00:00-05:00      1.0      5.0 2020-01-07 00:00:00-05:00
         2020-01-13 04:00:00-05:00      2.0      6.0 2020-01-11 00:00:00-05:00
         2020-01-15 04:00:00-05:00      3.0      7.0 2020-01-14 00:00:00-05:00
    NVDA 2020-01-02 04:00:00-05:00      0.0      4.0 2020-01-01 00:00:00-05:00
         2020-01-06 04:00:00-05:00      2.0      6.0 2020-01-05 00:00:00-05:00
         2020-01-13 04:00:00-05:00      3.0      7.0 2020-01-12 00:00:00-05:00
    AMZN 2020-01-07 04:00:00-05:00      0.0      4.0 2020-01-06 00:00:00-05:00
         2020-01-09 04:00:00-05:00      1.0      5.0 2020-01-08 14:30:00-05:00
         2020-01-10 04:00:00-05:00      2.0      6.0 2020-01-09 00:00:00-05:00
         2020-01-14 04:00:00-05:00      3.0      7.0 2020-01-13 00:00:00-05:00




```python
matched.compute()
```




                                    revcost  revenue                    fromix
    AAPL 2020-01-02 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-03 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-06 04:00:00-05:00      0.0      4.0 2020-01-03 00:00:00-05:00
         2020-01-07 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-08 04:00:00-05:00      1.0      5.0 2020-01-07 00:00:00-05:00
         2020-01-09 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-10 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-13 04:00:00-05:00      2.0      6.0 2020-01-11 00:00:00-05:00
         2020-01-14 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-15 04:00:00-05:00      3.0      7.0 2020-01-14 00:00:00-05:00
    NVDA 2020-01-02 04:00:00-05:00      0.0      4.0 2020-01-01 00:00:00-05:00
         2020-01-03 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-06 04:00:00-05:00      2.0      6.0 2020-01-05 00:00:00-05:00
         2020-01-07 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-08 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-09 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-10 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-13 04:00:00-05:00      3.0      7.0 2020-01-12 00:00:00-05:00
         2020-01-14 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-15 04:00:00-05:00      NaN      NaN                       NaT
    AMZN 2020-01-02 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-03 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-06 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-07 04:00:00-05:00      0.0      4.0 2020-01-06 00:00:00-05:00
         2020-01-08 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-09 04:00:00-05:00      1.0      5.0 2020-01-08 14:30:00-05:00
         2020-01-10 04:00:00-05:00      2.0      6.0 2020-01-09 00:00:00-05:00
         2020-01-13 04:00:00-05:00      NaN      NaN                       NaT
         2020-01-14 04:00:00-05:00      3.0      7.0 2020-01-13 00:00:00-05:00
         2020-01-15 04:00:00-05:00      NaN      NaN                       NaT




```python
closes = daily["close"]
```


```python
closes[["revcost", "revenue"]] = matched
```


```python
closes.compute()
```




                                    close  revcost  revenue
    AAPL 2020-01-02 04:00:00-05:00  75.30      NaN      NaN
         2020-01-03 04:00:00-05:00  74.33      NaN      NaN
         2020-01-06 04:00:00-05:00  74.91      0.0      4.0
         2020-01-07 04:00:00-05:00  73.15      NaN      NaN
         2020-01-08 04:00:00-05:00  76.00      1.0      5.0
         2020-01-09 04:00:00-05:00  77.60      NaN      NaN
         2020-01-10 04:00:00-05:00  77.67      NaN      NaN
         2020-01-13 04:00:00-05:00  79.70      2.0      6.0
         2020-01-14 04:00:00-05:00  77.94      NaN      NaN
         2020-01-15 04:00:00-05:00  78.05      3.0      7.0
    AMZN 2020-01-02 04:00:00-05:00  94.99      NaN      NaN
         2020-01-03 04:00:00-05:00  93.57      NaN      NaN
         2020-01-06 04:00:00-05:00  95.08      NaN      NaN
         2020-01-07 04:00:00-05:00  93.15      0.0      4.0
         2020-01-08 04:00:00-05:00  94.78      NaN      NaN
         2020-01-09 04:00:00-05:00  95.17      1.0      5.0
         2020-01-10 04:00:00-05:00  94.06      2.0      6.0
         2020-01-13 04:00:00-05:00  94.63      NaN      NaN
         2020-01-14 04:00:00-05:00  93.25      3.0      7.0
         2020-01-15 04:00:00-05:00  93.24      NaN      NaN
    NVDA 2020-01-02 04:00:00-05:00  60.01      0.0      4.0
         2020-01-03 04:00:00-05:00  59.02      NaN      NaN
         2020-01-06 04:00:00-05:00  59.35      2.0      6.0
         2020-01-07 04:00:00-05:00  57.88      NaN      NaN
         2020-01-08 04:00:00-05:00  60.25      NaN      NaN
         2020-01-09 04:00:00-05:00  60.93      NaN      NaN
         2020-01-10 04:00:00-05:00  61.08      NaN      NaN
         2020-01-13 04:00:00-05:00  63.20      3.0      7.0
         2020-01-14 04:00:00-05:00  61.68      NaN      NaN
         2020-01-15 04:00:00-05:00  61.54      NaN      NaN


