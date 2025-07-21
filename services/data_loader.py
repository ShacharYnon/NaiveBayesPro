import pandas as pd
import numpy as np


class DataLoader :

    def __init__(self ,path :str ,label_column :str):
        self.Path = path
        self.Label_column = label_column
        self.Data = None

    # @staticmethod
    def read_csv(self):
        try:
            self.Data = pd.read_csv(self.Path)
            return self.Data

        except Exception as e:
            print(f"ERROR from ReadCSV: {e}")

    # @staticmethod
    def read_sql(self):
        try:
            self.Data = pd.read_sql(self.Path)
            return self.Data

        except Exception as e:
            print(f"ERROR from ReadSql: {e}")

    # @staticmethod
    def read_json(self):
        try:
            self.Data = pd.read_json(self.Path)
            return self.Data

        except Exception as e:
            print(f"ERROR from ReadJson: {e}")

    # @staticmethod
    def read_excel(self):
        try:
            self.Data = pd.read_excel(self.Path)
            return self.Data

        except Exception as e:
            print(f"ERROR from ReadExcel: {e}")

    def split_test_train(self ,train_ratio = 0.7 ,shuffle=True):
        if self.Data is None:
            raise ValueError("Data not loaded. Please run read_csv() first.")

        if shuffle:
            self.Data = self.Data.sample(frac=1).reset_index(drop=True)

        train_size = int(len(self.Data)* train_ratio)
        train_data = self.Data.iloc[:train_size]
        test_data =self.Data.iloc[train_size:]

        return train_data ,test_data

    def get_features_and_labels(self):
        if self.Data is None:
            raise ValueError("Data not loaded. Please run read_csv() first.")

        features = self.Data.drop(columns=[self.Label_column])
        labels = self.Data[self.Label_column]
        return features ,labels

    def drot_column(self ,name_column:str |list[str] | None = None):
        if name_column is None:
            print("No column specified to drop.")
            return

        if isinstance(name_column ,str):
            name_column = [name_column]

        for column in name_column:
            if column in self.Data.columns:
                self.Data = self.Data.drop(columns=column)
                print(f"Column '{column}' dropped.")

            else:
                print(f"Column '{column}' not found in features.")