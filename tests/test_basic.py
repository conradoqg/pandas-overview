from pandas_overview import DataFrameSummary
import unittest
from random import shuffle
import numpy as np
import pandas as pd


class DataFrameSummaryTest(unittest.TestCase):
    """
    Test the new methods added by Alfonso R. Reyes.
    Dataframe has been expanded to show more columns of the same type.
    Needed for the summary.
    """
    def setUp(self):
        self.size = 1000
        missing = [np.nan] * (self.size // 10) + list(range(10)) * ((self.size - self.size // 10) // 10)
        shuffle(missing)

        self.types = [DataFrameSummary.TYPE_NUMERIC, DataFrameSummary.TYPE_BOOL,
                      DataFrameSummary.TYPE_CATEGORICAL, DataFrameSummary.TYPE_CONSTANT,
                      DataFrameSummary.TYPE_UNIQUE, DataFrameSummary.TYPE_DATE]

        self.columns = ['dbool1', 'dbool2',
                        'duniques1', 'duniques2',
                        'dcategoricals1', 'dcategoricals2',
                        'dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing',
                        'dconstant',
                        'ddates1', 'ddates2']

        self.df = pd.DataFrame(dict(
            dbool1=np.random.choice([0, 1], size=self.size),
            dbool2=np.random.choice(['a', 'b'], size=self.size),
            duniques1=['x{}'.format(i) for i in range(self.size)],
            duniques2=['y{}'.format(i) for i in range(self.size)],
            dcategoricals1=['a'.format(i) if i % 2 == 0 else
                            'b'.format(i) if i % 3 == 0 else
                            'c'.format(i) for i in range(self.size)],
            dcategoricals2=['x'.format(i) if i % 2 == 0 else
                            'y'.format(i) if i % 3 == 0 else
                            'z'.format(i) for i in range(self.size)],
            dnumerics1=range(self.size),
            dnumerics2=range(self.size, 2 * self.size),
            dnumerics3=list(range(self.size - self.size // 10)) + list(range(-self.size // 10, 0)),
            dmissing=missing,
            dconstant=['a'] * self.size,
            ddates1=pd.date_range('2010-01-01', periods=self.size, freq='1M'),
            ddates2=pd.date_range('2000-01-01', periods=self.size, freq='1W'),
        ))

        self.dfs = DataFrameSummary(self.df)

    def test__columns_stats(self):
        """
        Test the _columns_stats instance variable and the columns of the test dataframe.
        :return:
        """
        columns_stats = self.dfs._columns_stats
        print(type(columns_stats))
        self.assertIsInstance(columns_stats, pd.core.frame.DataFrame)
        expected = ['dbool1', 'dbool2', 'dcategoricals1', 'dcategoricals2', 'dconstant', 'ddates1', 'ddates2',
                    'dmissing', 'dnumerics1', 'dnumerics2', 'dnumerics3', 'duniques1', 'duniques2']
        result = columns_stats.columns.tolist()
        print(result)
        self.assertEqual(expected, result)

    def test__is_all_numeric_false(self):
        """
        Test that not all the columns provided in the list are "numeric".
        It must return "False"
        :return:
        """
        columns = ['dbool1', 'dbool2', 'dcategoricals', 'dconstant', 'ddates', 'dmissing',
                   'dnumerics1', 'dnumerics2', 'dnumerics3', 'duniques']
        result = self.dfs._is_all_numeric(columns)
        print(result)
        self.assertFalse(result)

    def test__is_all_numeric_true(self):
        """
        Test that all columns passed are "numeric".
        It must be "True"
        :return:
        """
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3']
        result = self.dfs._is_all_numeric(columns)
        print(result)
        self.assertTrue(result)

    def test__is_all_numeric_true_missing(self):
        """
        Numeric columns provided this time included NaNs.
        It muest be "True"
        :return:
        """
        #: includes missing nan column, which is numeric as well
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing']
        result = self.dfs._is_all_numeric(columns)
        print(result)
        self.assertTrue(result)

    def test_columns_of_type_numeric(self):
        """
        Test that a list of numeric columns matches the test dataframe
        :return:
        """
        expected = ['dmissing', 'dnumerics1', 'dnumerics2', 'dnumerics3']
        result = self.dfs.columns_of_type("numeric")
        print(result)
        print(self.dfs[result])
        self.assertTrue(expected == result)

    def test_columns_of_type_numeric_generic(self):
        """
        Test that all the columns returning are all of the same `numeric` type
        :return:
        """
        the_type = "numeric"
        columns = self.dfs.columns_of_type(the_type)
        frame = self.dfs[columns]
        print(frame)
        types = frame.ix['types']
        set_of_types = set(types.tolist())
        result = the_type in set_of_types
        print(result)
        self.assertTrue(result)

    def test_type_summary_numeric(self):
        """
        Test that the columns types reduce to a unique numeric value and matches.
        :return:
        """
        frame = self.dfs.type_summary('numeric')
        print(frame)
        result = self.dfs.TYPE_NUMERIC in set(frame.ix['types'])
        print(result)
        self.assertTrue(result)

    def test_columns_of_type_boolean(self):
        """
        Test that boolean columns match the type `bool`
        :return:
        """
        expected = ['dbool1', 'dbool2']
        result = self.dfs.columns_of_type("bool")
        print(result)
        self.assertTrue(expected == result)

    def test_show_dataframe_per_type(self):
        """
        Shows a column, one by one grouping by column type
        :return:
        """
        for column in self.types:
            print(column)
            columns = self.dfs.columns_of_type(column)
            # print(self.dfs[columns])
            list_of = columns
            for col in list_of:
                print(self.dfs[col])

    def test_columns_of_type_bool_generic(self):
        """
        This is an OLD behavior. Now corrected.
        There is a problem when the list of columns specified is not numeric: what returns when
        dfs[columns] is specified could be a list of the columns values.
        No what we are looking for.
        """
        the_type = "bool"
        columns = self.dfs.columns_of_type(the_type)
        print(columns)
        df = self.dfs[['dbool1', 'dbool2']]
        print(df)
        self.assertTrue(df.shape[1] == 2)

    def test_get_all_series_bool(self):
        """
        Test that boolean summary return the same number of rows.
        WIth the new behavior the number of rows must be 9 in the case of booleans
        :return:
        """
        list_of = ['dbool1', 'dbool2']
        for col in list_of:
            ser = self.dfs[col]
            print ser
            print ser.shape[0]
            self.assertTrue(ser.shape[0] == 9)

    def test_show_columns_types(self):
        """
        Test that the columns in the test dataframe is a subset of the class variable "types"
        :return:
        """
        self.assertTrue(set(self.dfs.columns_types.index).issubset(self.dfs.types))

    def test__is_type_the_same_bool(self):
        """
        Test that the columns passed are of the same type
        :return:
        """
        columns = ['dbool1', 'dbool2']
        list_of_types = self.dfs._is_type_the_same(columns)
        self.assertTrue(list_of_types)

    def test__is_type_the_same_many_false(self):
        """
        Tests that the columns passed are NOT all of the same type
        :return:
        """
        columns = ['dbool1', 'dbool2', 'dnumerics1']
        list_of_types = self.dfs._is_type_the_same(columns)
        self.assertFalse(list_of_types)

    def test__is_type_the_same_numeric(self):
        """
        Test that the columns passed are all the same
        :return:
        """
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing']
        list_of_types = self.dfs._is_type_the_same(columns)
        self.assertTrue(list_of_types)

    def test_get_all_the_same_unique(self):
        """
        Test that the unique columns passed are all unique
        :return:
        """
        columns = ['duniques1', 'duniques2']
        self.assertTrue(set(self.dfs[columns].loc['types'].tolist()) == {'unique'})

    def test_get_all_the_same_numeric(self):
        """
        Test that all the numeric columns are all numeric
        """
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing']
        self.assertTrue(set(self.dfs[columns].loc['types'].tolist()) == {'numeric'})

    def test_get_all_the_same_categorical(self):
        """
        Tests that all categorical columns reduce to `categorical`
        :return:
        """
        columns = ['dcategoricals1', 'dcategoricals2']
        self.assertTrue(set(self.dfs[columns].loc['types'].tolist()) == {'categorical'})

    def test_get_all_the_same_dates(self):
        """
        Test that all the ``date columns reduce to a unique type `date`
        :return:
        """
        columns = ['ddates1', 'ddates2']
        self.assertTrue(set(self.dfs[columns].loc['types'].tolist()) == {'date'})
