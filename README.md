# pandas_overview
> A tool that goes deeper than [pandas](http://pandas.pydata.org/) dataframe's describe function.

The module contains `DataFrameSummary` and a `DataFrameOverview` objects. 

The `DataFrameSummary` is an extended version of pandas' `describe()` method with:

* **methods**
  * `summary(columns=None)`: extends the `describe()` function with detailed info:
    * count, mean, std, min, 25%, 50%, 75%, max, counts, uniques, mising, missing %, type
  * `dfs.type_summary(type=None)`: summary for a specific type, it has a more specialized info:
    * `numeric`: iqr, kurtosis, skewness, sum, mad, cv, zeros_num, zeros %, deviating of mean, deviating of mean %
    * `constant`: top row
    * `categorical`: values, top row
    * `bool`: left value, left value %, right value, right value %
    * `unique`: is not that unique :/
* **properties**
  * `dfs.columns_types`: a count of the types of columns
  * `dfs.types`: internal list of supported types  
  * `dfs[column]`: more in depth summary of the column

The `DataFrameOverview` displays an overview of the data, including:

* **methods**
  * `overview(dfs, first_level=None)`: Data overview showing:
    * Summary
    * Columns
    * Head
    * Tail
    * Correlations
    * Histogram for numeric columns
    * Range for unique and date columns
    * Values for constant columns
 

# Installation (Not published yet)
The module can be easily installed with pip:

```bash
> pip install pandas-overview
```

This module depends on `numpy` and `pandas`. Optionally you can get also some nice visualisations if you have `matplotlib` installed.

# Tests
To run the tests, execute the command `python setup.py test`.

# Usage
For detailed information check this [this](notebooks/01%20-%20GAP%20network%20scan.ipynb) and [this](notebooks/02%20-%20Prosper%20multiwell%20scan.ipynb) out.

# Origins
This is a forked version of the original [pandas-summary](https://github.com/mouradmourafiq/pandas-summary) plus great additions from [pandas-summary-master](https://github.com/AlfonsoRReyes/pandas-summary-master)

## Contribution and License Agreement

If you contribute code to this project, you are implicitly allowing your code
to be distributed under the MIT license. You are also implicitly verifying that
all code is your original work.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/conradoqg/pandas-overview/blob/master/LICENSE)