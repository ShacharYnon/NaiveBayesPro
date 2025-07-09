import pandas as pd
import numpy as np


class DataLoader :

    def __init__(self ,path :str ,label_column :str):
        self.Path = path
        self.Label_column = label_column
        self.Data = None

    def read_csv(self ):
        try:
            self.Data = pd.read_csv(self.Path)
            return self.Data

        except Exception as e:
            print(f"ERROR from ReadCSV: {e}")

    def split_test_train(self ,train_ratio = 0.7 ,shuffle=True):
        if self.Data is None:
            raise ValueError("Data not loaded. Please run read_csv() first.")

        if shuffle:
            self.Data = self.Data.sample(frac=1).reset_index(drop=True)

            train_size = int(len(self.Data)* train_ratio)
            train_data = self.Data.iloc[:train_size]
            test_data =self.Data.iloc[train_size:]

        return train_data ,test_data

