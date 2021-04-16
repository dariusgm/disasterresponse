import pandas as pd
import os
from sqlalchemy import create_engine

class Constant:
    @staticmethod
    def table_name() -> str:
        return 'etl'

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
        len(df) - len(df.drop_duplicates())

        engine = create_engine(f'sqlite:///{self.sql_path}')
        df.to_sql(Constant.table_name(), engine, index=False)

    def run(self):
        messages_df, categories_df = self.__extract()
        df = self.__transform(messages_df, categories_df)
        self.__load(df)

def main(messages_path, categories_path, sql_path):
    etl_pipeline = ETLPipeline(messages_path, categories_path, sql_path)
    etl_pipeline.run()

if __name__ == '__main__':
    messages_path = os.path.join("data", "messages.csv")
    categories_path = os.path.join("data", "categories.csv")
    sql_path = "etl.db"
    main(messages_path, categories_path, sql_path)



