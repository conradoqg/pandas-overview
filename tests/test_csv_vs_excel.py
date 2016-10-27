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
        This test shows that the column name types are not the same
        :return:
        """
        xdf_columns = self.xdf.columns.tolist()
        cdf_columns = self.cdf.columns.tolist()
        for x, c in zip(xdf_columns, cdf_columns):
            # print(type(x), type(c))
            self.assertFalse(type(x) == type(c))

    def test_column_excel_is_not_instance_of_str(self):
        xdf_columns = self.xdf.columns.tolist()
        for column in xdf_columns:
            self.assertFalse(isinstance(column, str))

    def test_column_csv_is_instance_of_str(self):
        cdf_columns = self.cdf.columns.tolist()
        for column in cdf_columns:
            self.assertTrue(isinstance(column, str))

    def test_pandas_summary_on_excel_df(self):
        #: fixme: this returns error
        xdfs = DataFrameSummary(self.xdf)
        print(xdfs.get_numeric_summary())

    def test_pandas_summary_on_csv_df(self):
        #: this works great!
        cdfs = DataFrameSummary(self.cdf)
        print(cdfs.get_numeric_summary())

    def test_clean_column_on_excel(self):
        xdfs = DataFrameSummary(self.xdf)
        xdf_columns = self.xdf.columns.tolist()

        print(xdfs._clean_column(xdf_columns[0]))

        for x in xdf_columns:
            # print(xdfs._clean_column(x))
            self.assertTrue(xdfs._clean_column(x))

    def test_get_list_of_type(self):
        xdfs = DataFrameSummary(self.xdf)
        the_type = "numeric"
        columns = xdfs._get_list_of_type(the_type)
        print(columns)
        # xdfs.get_numeric_summary()
        cols = [x.encode('ascii') for x in columns]
        print(cols)


