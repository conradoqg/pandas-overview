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
        display(Markdown("Data shape: " + str(dfs.df.shape)))
        display(Markdown(markdown_level + "## Columns"))
        for column_type in dfs.types:
            column_list = dfs._get_list_of_type(column_type)
            if column_list:
                display(Markdown("**"+column_type+":** " + str(column_list)))
            else:
                display(Markdown("**"+column_type+":** None"))
        
        display(Markdown(markdown_level + "# Head"))
        display(dfs.df.head())
        display(Markdown(markdown_level + "# Tail"))
        display(dfs.df.tail())
        display(Markdown(markdown_level + "# Correlations"))
        with np.errstate(invalid='ignore'):
            corr = dfs.df.corr()
            display(corr.style.background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1).set_properties(
                **{'max-width': '80px', 'font-size': '10pt'}).set_precision(2))
        if dfs._get_list_of_type("numeric"):
            display(Markdown(markdown_level + "# Histogram for numeric columns"))
            dfs.df[dfs._get_list_of_type("numeric")].hist(
                bins='auto', figsize=(20, 15))
            plt.show()
        if dfs._get_list_of_type("categorical"):
            max_rows = 20
            display(Markdown(markdown_level + "# Value counts for categorical columns"))
            display(DataFrameOverview.get_summary_value_counts(dfs, max_rows))
            display(HTML("<small>Limited to " +  str(max_rows) + " rows</small>"))
        if dfs._get_list_of_type("unique") or dfs._get_list_of_type("date"):
            display(Markdown(markdown_level + "# Range for unique and date columns"))            
            display(DataFrameOverview.get_range(dfs))
        if dfs._get_list_of_type("constant"):
            display(Markdown(markdown_level + "# Values for constant columns"))            
            display(DataFrameOverview.get_constant(dfs))

        # Restore column limitation
        pd.set_option('display.max_columns', max_columns)

    @staticmethod
    def get_summary_value_counts(dfs, max_rows=20):
        frames = []
        dfs.df[dfs._get_list_of_type("categorical")].apply(
            lambda X: frames.append(DataFrameOverview.limit_dataframe_rows(X, 20)))
        categorical_counts = pd.concat(frames, axis=1)
        categorical_counts.drop('index', axis=1, inplace=True)
        return categorical_counts

    @staticmethod
    def limit_dataframe_rows(filtered_dataframe, max_rows=20):
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
    def get_range(dfs):
        frames = []
        dfs.df[dfs._get_list_of_type("unique") + dfs._get_list_of_type("date")].apply(lambda X: frames.append([X.name, X.min(), X.max()]))    
        return pd.DataFrame(frames, columns=['Column', 'Max', 'Min'])

    @staticmethod
    def get_constant(dfs):
        frames = []
        dfs.df[dfs._get_list_of_type("constant")].apply(lambda X: frames.append([X.name, X.min()]))    
        return pd.DataFrame(frames, columns=['Column', 'Value'])
