---
layout: post
title: Timeseries-Missing values in time series
date: 2020-01-10
categories: Timeseries
tags: [Timeseries, Missing values]
---

![header image](https://images.unsplash.com/photo-1519297859939-c875d362d194?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=872&q=80){: w="700" h="400" }

## Introduction

<p>Missing values are common in real world data . we need to fill the missing data before feeding it to the model. In time series sequence the missing value can be be a random data or the sequence of data it occur due to incorrect reading or missing reading for that timesteps.one should pay attention while filling the missing  data in time series because it should be depend on time weather data the summer and winter has the extremely different temperature and also in seasonal like rainfall data you can expect minimum or no rain in summer and more rain in rainy season depends on the country. Then make sure your data to be realistic with respect to time.
Let's consider the data with average daily temperature for year 2000.</p>

<p>Interpolation is the process of using points with known values or sample points to estimate values at other unknown points.</p>
```python
import pandas as pd
from datetime import datetime
import numpy as np

#importing the txt data file
data = pd.read_csv('/avg_temp.txt', sep=" ", header=None)

#Assign the column names
data.columns = ["Date","average_temperature"]

#Display missing values count in the dataset
data.isna().sum()

````
<p>there are 40 missing values in the dataset .you noticed that Date column is of object type, before filling the missing values we have to convert the date column to datetime datatype and change it as the index for time series data.</p>
<p>In pandas, a single point in time is represented as a Timestamp. We can use the to_datetime() function to create Timestamps from strings in a wide variety of date/time formats.</p>
<p>If we supply a list or array of strings as input to to_datetime(), it returns a sequence of date/time values in a DatetimeIndex object, which is the core data structure that powers much of pandas time series functionality.</p>
```python
# converting the date column to datetime
data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y"))
data.dtypes
````

<p>Now Date column is changed to datetime64[ns] data type indicates that the underlying data is stored as 64-bit integers, in units of nanoseconds (ns). This data structure allows pandas to compactly store large sequences of date/time values and efficiently perform vectorized operations using NumPy datetime64 arrays.</p>
```python
# converting date column as index
data=data.set_index('Date')
print("minimum data is :",data.index.min().date())
print("maximum date is :",data.index.max().date())
```
## Visualizing the time series data
