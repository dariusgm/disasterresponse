import sys


import pandas as pd
import os
from sqlalchemy import create_engine
from constant import Constat

class ETLPipeline:
    def __init__(self, messages_path: str, categories_path: str, sql_path: str):
        self.messages_path = messages_path
        self.categories_path = categories_path
        self.sql_path = sql_path

    def __count_duplucates(self, df: pd.DataFrame):
        pass

    def __transform_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        # create a dataframe of the 36 individual category columns
        categories = df['categories'].str.split(expand=True, pat=";")

        # select the first row of the categories dataframe
        row = categories.iloc[0]

        # use this row to extract a list of new column names for categories.
        # one way is to apply a lambda function that takes everything 
        # up to the second to last character of each string with slicing
        category_colnames = row.apply(lambda x: x.split("-")[0])

        # rename the columns of `categories`
        categories.columns = category_colnames

        # Convert category values to just numbers 0 or 1
        for column in categories:
            # set each value to be the last character of the string
            # convert column from string to numeric
            categories[column] = categories[column].astype(str).apply(lambda x: int(x[-1]))

        return categories

    def __extract(self):
        return [
            pd.read_csv(self.messages_path), 
            pd.read_csv(self.categories_path)
        ]

    def __transform(self, categories: pd.DataFrame, messages: pd.DataFrame):
        df =  pd.concat([categories, messages], axis=1)
        categories = self.__transform_categories(df)

        df = df.drop(columns='categories')
        return pd.concat([df, categories], axis=1)

    def __load(self, df: pd.DataFrame):
        # check number of duplicates
        len(df) - len(df.drop_duplicates())

        # drop duplicates
        df = df.drop_duplicates()

        # check number of duplicates
        print("length without duplicates: {}").format(len(df) - len(df.drop_duplicates()))

        engine = create_engine(f'sqlite:///{self.sql_path}')
        df.to_sql(Constant.table_name(), engine, index=False)

    def run(self):
        messages_df, categories_df = self.__extract()
        df = self.__transform(messages_df, categories_df)
        self.__load(df)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        etl_pipeline = ETLPipeline(messages_filepath, categories_filepath, database_filepath)
        etl_pipeline.run()    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()