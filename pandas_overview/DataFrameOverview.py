import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from IPython.display import display, HTML, Markdown


class DataFrameOverview:
    def overview(dfs, first_level = 1):
        markdown_level = "#" * first_level

        # Remove column limitation, it's necessary to get a real overview of the data
        max_columns = pd.get_option('display.max_columns')
        pd.set_option('display.max_columns', None)

        display(Markdown(markdown_level +" Data overview"))
        display(Markdown(markdown_level + "# Summary"))
        display(dfs.summary())
        display(Markdown("Data shape: " + str(dfs._df.shape)))
        display(Markdown(markdown_level + "## Columns"))
        for column_type in dfs.types:
            column_list = dfs.columns_of_type(column_type)
            if column_list:
                display(Markdown("**"+column_type+":** " + str(column_list)))
            else:
                display(Markdown("**"+column_type+":** None"))
        
        display(Markdown(markdown_level + "# Head"))
        display(dfs._df.head())
        display(Markdown(markdown_level + "# Tail"))
        display(dfs._df.tail())
        display(Markdown(markdown_level + "# Correlations"))
        with np.errstate(invalid='ignore'):
            corr = dfs._df.corr()
            display(corr.style.background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1).set_properties(
                **{'max-width': '80px', 'font-size': '10pt'}).set_precision(2))
        if dfs.columns_of_type("numeric"):
            display(Markdown(markdown_level + "# Histogram for numeric columns"))
            dfs._df[dfs.columns_of_type("numeric")].hist(
                bins='auto', figsize=(20, 15))
            plt.show()
        if dfs.columns_of_type("categorical"):
            max_rows = 20
            display(Markdown(markdown_level + "# Value counts for categorical columns"))
            display(DataFrameOverview._get_summary_value_counts(dfs, max_rows))
            display(HTML("<small>Limited to " +  str(max_rows) + " rows</small>"))
        if dfs.columns_of_type("unique") or dfs.columns_of_type("date"):
            display(Markdown(markdown_level + "# Range for unique and date columns"))            
            display(DataFrameOverview._get_range(dfs))
        if dfs.columns_of_type("constant"):
            display(Markdown(markdown_level + "# Values for constant columns"))            
            display(DataFrameOverview._get_constant(dfs))

        # Restore column limitation
        pd.set_option('display.max_columns', max_columns)

    @staticmethod
    def _get_summary_value_counts(dfs, max_rows=20):
        frames = []
        dfs._df[dfs.columns_of_type("categorical")].apply(
            lambda X: frames.append(DataFrameOverview._limit_dataframe_rows(X, 20)))
        categorical_counts = pd.concat(frames, axis=1)
        categorical_counts.drop('index', axis=1, inplace=True)
        return categorical_counts

    @staticmethod
    def _limit_dataframe_rows(filtered_dataframe, max_rows=20):
        df = filtered_dataframe.value_counts().to_frame()
        number_of_columns = math.floor(max_rows / 2)
        df.reset_index(level=0, inplace=True)
        df.rename(columns={
                  'index': df.columns[1], df.columns[1]: df.columns[1] + ' Count'}, inplace=True)
        head_rows = df[:number_of_columns]
        tail_rows = df[-number_of_columns:]
        value_count_dataframe = pd.concat(
            [head_rows, pd.DataFrame([['...', '...']], columns=df.columns), tail_rows])
        value_count_dataframe.reset_index(level=0, inplace=True)
        return value_count_dataframe

    @staticmethod
    def _get_range(dfs):
        frames = []
        dfs._df[dfs.columns_of_type("unique") + dfs.columns_of_type("date")].apply(lambda X: frames.append([X.name, X.min(), X.max()]))    
        return pd.DataFrame(frames, columns=['Column', 'Max', 'Min'])

    @staticmethod
    def _get_constant(dfs):
        frames = []
        dfs._df[dfs.columns_of_type("constant")].apply(lambda X: frames.append([X.name, X.min()]))    
        return pd.DataFrame(frames, columns=['Column', 'Value'])
