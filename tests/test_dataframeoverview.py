from __future__ import division

from random import shuffle
import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal

from pandas_overview import DataFrameSummary


class DataFrameSummaryTest(unittest.TestCase):
    #: fixed some TYPE comparisons on 20161026
    def setUp(self):
        self.size = 1000
        missing = [np.nan] * (self.size // 10) + list(range(10)) * ((self.size - self.size // 10) // 10)
        shuffle(missing)

        self.types = [DataFrameSummary.TYPE_NUMERIC, DataFrameSummary.TYPE_BOOL,
                      DataFrameSummary.TYPE_CATEGORICAL, DataFrameSummary.TYPE_CONSTANT,
                      DataFrameSummary.TYPE_UNIQUE, DataFrameSummary.TYPE_DATE]

        self.columns = ['dbool1', 'dbool2', 'duniques', 'dcategoricals', 'dnumerics1', 'dnumerics2',
                        'dnumerics3', 'dmissing', 'dconstant', 'ddates']

        self.df = pd.DataFrame(dict(
            dbool1=np.random.choice([0, 1], size=self.size),
            dbool2=np.random.choice(['a', 'b'], size=self.size),
            duniques=['x{}'.format(i) for i in range(self.size)],
            dcategoricals=['a'.format(i) if i % 2 == 0 else
                           'b'.format(i) if i % 3 == 0 else
                           'c'.format(i) for i in range(self.size)],
            dnumerics1=range(self.size),
            dnumerics2=range(self.size, 2 * self.size),
            dnumerics3=list(range(self.size - self.size // 10)) + list(range(-self.size // 10, 0)),
            dmissing=missing,
            dconstant=['a'] * self.size,
            ddates=pd.date_range('2010-01-01', periods=self.size, freq='1M')))

        self.dfs = DataFrameSummary(self.df)

    def test_get_columns_works_as_expected(self):
        assert len(self.dfs.get_columns(self.df, DataFrameSummary.ALL)) == 10

        assert len(self.dfs.get_columns(self.df,
                                        DataFrameSummary.INCLUDE,
                                        ['dnumerics1', 'dnumerics2', 'dnumerics3'])) == 3

        assert len(self.dfs.get_columns(self.df,
                                        DataFrameSummary.EXCLUDE,
                                        ['dnumerics1', 'dnumerics2', 'dnumerics3'])) == 7

    def test_column_types_works_as_expected(self):
        expected = pd.Series(index=self.types, data=[4, 2, 1, 1, 1, 1], name='types')
        assert_series_equal(self.dfs.columns_types[self.types], expected[self.types])

    def test_column_stats_works_as_expected(self):
        column_stats = self.dfs.columns_stats
        self.assertTupleEqual(column_stats.shape, (5, 10))

        # counts
        expected = pd.Series(index=self.columns,
                             data=self.size,
                             name='counts',
                             dtype='object')
        expected['dmissing'] -= 100
        assert_series_equal(column_stats[self.columns].loc['counts'],
                            expected[self.columns])

        # uniques
        expected = pd.Series(index=self.columns,
                             data=self.size,
                             name='uniques',
                             dtype='object')
        expected[['dbool1', 'dbool2']] = 2
        expected['dcategoricals'] = 3
        expected['dconstant'] = 1
        expected['dmissing'] = 10
        print(column_stats[self.columns].loc['uniques'])
        print(expected[self.columns])
        assert_series_equal(column_stats[self.columns].loc['uniques'],
                            expected[self.columns].astype('object'))

        # missing
        expected = pd.Series(index=self.columns,
                             data=0,
                             name='missing',
                             dtype='object')
        expected[['dmissing']] = 100
        assert_series_equal(column_stats[self.columns].loc['missing'],
                            expected[self.columns].astype('object'))

        # missing_perc
        expected = pd.Series(index=self.columns,
                             data=['0%'],
                             name='missing_perc',
                             dtype='object')

        expected[['dmissing']] = '10%'
        assert_series_equal(column_stats[self.columns].loc['missing_perc'],
                            expected[self.columns].astype('object'))

        # types
        expected = pd.Series(index=self.columns,
                             data=[np.nan],
                             name='types',
                             dtype='object')

        expected[['dbool1', 'dbool2']] = DataFrameSummary.TYPE_BOOL
        expected[['dcategoricals']] = DataFrameSummary.TYPE_CATEGORICAL
        expected[['dconstant']] = DataFrameSummary.TYPE_CONSTANT
        expected[['ddates']] = DataFrameSummary.TYPE_DATE
        expected[['duniques']] = DataFrameSummary.TYPE_UNIQUE
        expected[['dnumerics1', 'dnumerics2',
                  'dnumerics3', 'dmissing']] = DataFrameSummary.TYPE_NUMERIC
        assert_series_equal(column_stats[self.columns].loc['types'],
                            expected[self.columns].astype('object'))

    def test_numer_format_works_as_expected(self):
        float_nums = [(123.123, '123.12'),
                      (123.1243453, '123.12'),
                      (213213213.123, '213,213,213.12')]
        int_nums = [(213214, '213,214'),
                    (123213.00, '123,213')]

        for num, expected in float_nums:
            self.assertEqual(DataFrameSummary._number_format(num), expected)

        for num, expected in int_nums:
            self.assertEqual(DataFrameSummary._number_format(num), expected)

    def test_get_perc_works_as_expected(self):
        float_nums = [(0.123, '12.30%'),
                      (3.1243453, '312.43%'),
                      (213.12312, '21,312.31%')]

        int_nums = [(0.14, '14%'),
                    (1.300, '130%')]

        for num, expected in float_nums:
            self.assertEqual(DataFrameSummary._percent(num), expected)

        for num, expected in int_nums:
            self.assertEqual(DataFrameSummary._percent(num), expected)

    def test_uniques_summary(self):
        expected = pd.Series(index=['counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=[self.size, self.size, 0, '0%', DataFrameSummary.TYPE_UNIQUE],
                             name='duniques',
                             dtype=object)
        assert_series_equal(self.dfs['duniques'],
                            expected)

    def test_constant_summary(self):
        #: fixed on 20161026
        expected = pd.Series(index=['top', 'counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=['a: 1000', self.size, 1, 0, '0%', DataFrameSummary.TYPE_CONSTANT],
                             name='dconstant', dtype=object
                             )
        print(expected)
        print(self.dfs['dconstant'])
        assert_series_equal(self.dfs['dconstant'], expected)

    def test_bool1_summary(self):
        count_values = self.df['dbool1'].value_counts()
        total_count = self.df['dbool1'].count()
        count0 = count_values[0]
        count1 = count_values[1]
        perc0 = DataFrameSummary._percent(count0 / total_count)
        perc1 = DataFrameSummary._percent(count1 / total_count)
        expected = pd.Series(index=['"0" count', '"0" perc', '"1" count', '"1" perc',
                                    'counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=[str(count0), perc0, str(count1), perc1,
                                   self.size, 2, 0, '0%', DataFrameSummary.TYPE_BOOL],
                             name='dbool1',
                             dtype=object)

        assert_series_equal(self.dfs['dbool1'], expected)

    def test_bool2_summary(self):
        count_values = self.df['dbool2'].value_counts()
        total_count = self.df['dbool2'].count()
        count0 = count_values['a']
        count1 = count_values['b']
        perc0 = DataFrameSummary._percent(count0 / total_count)
        perc1 = DataFrameSummary._percent(count1 / total_count)
        expected = pd.Series(index=['"a" count', '"a" perc', '"b" count', '"b" perc',
                                    'counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=[str(count0), perc0, str(count1), perc1,
                                   self.size, 2, 0, '0%', DataFrameSummary.TYPE_BOOL],
                             name='dbool2',
                             dtype=object)

        assert_series_equal(self.dfs['dbool2'], expected)
        print(expected)

    def test_categorical_summary(self):
        #: fixed on 20161026
        expected = pd.Series(index=['cats', 'top', 'counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=[{'a', 'c', 'b'}, 'a: 500',
                                   self.size, 3, 0, '0%', DataFrameSummary.TYPE_CATEGORICAL],
                             name='dcategoricals',
                             dtype=object)

        assert_series_equal(self.dfs['dcategoricals'], expected)
        print(self.dfs['dcategoricals'])
        print(expected)

    def test_dates_summary(self):
        #: fixed on 20161026
        dmin = self.df['ddates'].min()
        dmax = self.df['ddates'].max()
        freq = pd.infer_freq(self.df['ddates'])

        expected = pd.Series(index=['freq', 'max', 'min', 'range',
                                    'counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=[freq, dmax, dmin, dmax - dmin,
                                   self.size, self.size, 0, '0%', DataFrameSummary.TYPE_DATE],
                             name='ddates',
                             dtype=object)

        assert_series_equal(self.dfs['ddates'], expected)
        print(self.dfs['ddates'])
        print(expected)

    def test_numerics_summary(self):
        #: fixed on 20161026
        num1 = self.df['dnumerics1']
        dm, dmp = self.dfs._get_deviation_of_mean(num1)
        dam, damp = self.dfs._get_median_absolute_deviation(num1)

        #: new expected variable with `top_correlations` removed
        expected = pd.Series(index=['mean', 'std', 'variance', 'min', 'max',
                                    '5%', '25%', '50%',
                                    '75%', '95%', 'iqr',
                                    'kurtosis', 'skewness', 'sum', 'mad',
                                    'cv',
                                    'zeros_num',
                                    'zeros_perc',
                                    'deviating_of_mean', 'deviating_of_mean_perc',
                                    'deviating_of_median', 'deviating_of_median_perc',
                                    # 'top_correlations',                               #: removing top_correlations
                                    'counts', 'uniques', 'missing', 'missing_perc', 'types'],
                             data=[num1.mean(), num1.std(), num1.var(), num1.min(), num1.max(),
                                   num1.quantile(0.05), num1.quantile(0.25), num1.quantile(0.5),
                                   num1.quantile(0.75), num1.quantile(0.95), num1.quantile(0.75) - num1.quantile(0.25),
                                   num1.kurt(), num1.skew(), num1.sum(), num1.mad(),
                                   num1.std() / num1.mean() if num1.mean() else np.nan,
                                   self.size - np.count_nonzero(num1),
                                   DataFrameSummary._percent(
                                       (self.size - np.count_nonzero(num1)) / self.size),
                                   dm, dmp,
                                   dam, damp,
                                   # 'dnumerics2: 100%',                                #: removing top_correlations
                                   self.size, self.size, 0, '0%', DataFrameSummary.TYPE_NUMERIC],
                             name='dnumerics1',
                             dtype=object)

        print(self.dfs['dnumerics1'])
        print(expected)
        assert_series_equal(self.dfs['dnumerics1'], expected)
