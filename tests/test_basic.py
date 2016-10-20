# from ptech.ext.pandas_summary import DataFrameSummary
from pandas_summary import DataFrameSummary
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

    def test_columns_stats(self):
        columns_stats = self.dfs.columns_stats
        print(type(columns_stats))
        self.assertIsInstance(columns_stats, pd.core.frame.DataFrame)
        expected = ['dbool1', 'dbool2', 'dcategoricals1', 'dcategoricals2', 'dconstant', 'ddates1', 'ddates2',
                    'dmissing', 'dnumerics1', 'dnumerics2', 'dnumerics3', 'duniques1', 'duniques2']
        result = columns_stats.columns.tolist()
        print(result)
        self.assertEqual(expected, result)

    def test__is_all_numeric_false(self):
        columns = ['dbool1', 'dbool2', 'dcategoricals', 'dconstant', 'ddates', 'dmissing',
                   'dnumerics1', 'dnumerics2', 'dnumerics3', 'duniques']
        result = self.dfs._is_all_numeric(columns)
        print(result)
        self.assertFalse(result)

    def test__is_all_numeric_true(self):
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3']
        result = self.dfs._is_all_numeric(columns)
        print(result)
        self.assertTrue(result)

    def test__is_all_numeric_true_missing(self):
        #: includes missing nan column, which is numeric as well
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing']
        result = self.dfs._is_all_numeric(columns)
        print(result)
        self.assertTrue(result)

    def test__get_list_of_type_numeric(self):
        expected = ['dmissing', 'dnumerics1', 'dnumerics2', 'dnumerics3']
        result = self.dfs._get_list_of_type("numeric")
        print(result)
        print(self.dfs[result])
        self.assertTrue(expected == result)

    def test__get_list_of_type_numeric_generic(self):
        the_type = "numeric"
        columns = self.dfs._get_list_of_type(the_type)
        frame = self.dfs[columns]
        print(frame)
        types = frame.ix['types']
        set_of_types = set(types.tolist())
        result = the_type in set_of_types
        print(result)
        self.assertTrue(result)

    def test_get_numeric_summary(self):
        frame = self.dfs.get_numeric_summary()
        print(frame)
        result = self.dfs.TYPE_NUMERIC in set(frame.ix['types'])
        print(result)
        self.assertTrue(result)

    def test__get_list_of_type_boolean(self):
        expected = ['dbool1', 'dbool2']
        result = self.dfs._get_list_of_type("bool")
        print(result)
        self.assertTrue(expected == result)

    def test_show_dataframe_per_type(self):
        """
        Shows a column, one by one grouping by column type
        :return:
        """
        for column in self.types:
            print(column)
            columns = self.dfs._get_list_of_type(column)
            # print(self.dfs[columns])
            list_of = columns
            for col in list_of:
                print(self.dfs[col])

    def test__get_list_of_type_bool_generic(self):
        """
        There is a problem when the list of columns specified is not numeric: what returns when
        dfs[columns] is specified could be a list of the columns values.
        No waht we are looking for.
        """
        the_type = "bool"
        columns = self.dfs._get_list_of_type(the_type)
        print(columns)
        print(self.dfs['dbool2'])
        # frame = self.dfs[columns]
        # print(frame)
        # types = frame.ix['types']
        # set_of_types = set(types.tolist())
        # result = the_type in set_of_types
        # print(result)
        # self.assertTrue(result)

    def test_get_all_bool(self):
        list_of = ['dbool1', 'dbool2']
        for col in list_of:
            print(self.dfs[col])

    def test_show_columns_types(self):
        print(self.dfs.columns_types)

    def test_show_all_types(self):
        for t in self.dfs.types:
            print(t)

    def test__is_type_the_same_bool(self):
        columns = ['dbool1', 'dbool2']
        list_of_types = self.dfs._is_type_the_same(columns)
        print(list_of_types)

    def test__is_type_the_same_many(self):
        columns = ['dbool1', 'dbool2', 'dnumerics1']
        list_of_types = self.dfs._is_type_the_same(columns)
        print(list_of_types)

    def test__is_type_the_same_numeric(self):
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing']
        list_of_types = self.dfs._is_type_the_same(columns)
        print(list_of_types)

    def test_get_all_the_same_bool(self):
        columns = ['dbool1', 'dbool2']
        print(self.dfs[columns])

    def test_get_all_the_same_unique(self):
        columns = ['duniques1', 'duniques2']
        print(self.dfs[columns])

    def test_get_all_the_same_numeric(self):
        columns = ['dnumerics1', 'dnumerics2', 'dnumerics3', 'dmissing']
        print(self.dfs[columns])

    def test_get_all_the_same_categorical(self):
        columns = ['dcategoricals1', 'dcategoricals2']
        print(self.dfs[columns])

    def test_get_all_the_same_dates(self):
        columns = ['ddates1', 'ddates2']
        print(self.dfs[columns])