from __future__ import division, print_function

from random import shuffle
import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal

from pandas_summary import DataFrameSummary


class ExcelVsCSVTest(unittest.TestCase):
    #: fixed some TYPE comparisons on 20161026
    def setUp(self):
        self.xdf = pd.read_excel("../data/gap_glop_60_dataset.xlsx")
        self.cdf = pd.read_csv("../data/gap_glop_60_dataset.csv")
        pass

    def test_excel_csv_same_shape(self):
        print(self.xdf.shape)
        print(self.cdf.shape)
        self.assertEqual(self.xdf.shape, self.cdf.shape)

    def test_same_dtypes(self):
        xdf_dtypes = self.xdf.dtypes
        cdf_dtypes = self.cdf.dtypes
        print(type(xdf_dtypes))
        # print(self.xdf.dtypes)
        assert_series_equal(xdf_dtypes, cdf_dtypes)

    def test_same_columns(self):
        xdf_columns = self.xdf.columns.tolist()
        cdf_columns = self.cdf.columns.tolist()
        self.assertListEqual(xdf_columns, cdf_columns)
        print(xdf_columns)
        print(cdf_columns)

    def test_column_names_same_type_false(self):
        """
        Excel import the columns as unicode meanwhile CSV import as str()
        :return:
        """
        xdf_columns = self.xdf.columns.tolist()
        cdf_columns = self.cdf.columns.tolist()
        for x, c in zip(xdf_columns, cdf_columns):
            print(type(x), type(c))
            self.assertFalse(type(x) == type(c))
