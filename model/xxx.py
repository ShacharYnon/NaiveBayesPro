from services.data_loader import DataLoader
from services import *
import pandas as pd
import numpy as np

class Classifier:

    def __init__(self ,features ,labels):
        self.Features = features
        self.Labels = labels
        self.Sub_features = {}


    def calculate_priors(self, features):
        total = len(self.Labels)
        priors = {}
        for label in self.Labels.unique():
            priors[label] = len(self.Labels[self.Labels == label]) / total
        return priors


    def predict(self, sample):
        best_label = None
        best_log_prob = float("-inf")
        priors = self.calculate_priors(self.Labels)
        for label in self.Sub_features:
            log_prob = np.log(priors[label])
            for feature, value in sample.items():
                prob = self.Sub_features[label].get(feature, {}).get(value, 1e-6)
                log_prob += np.log(prob)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_label = label
        return best_label




    def fit(self):
        df_features = self.Features
        df_labels = self.Labels
        for label in df_labels.unique():
            subset = df_features[df_labels == label]
            self.Sub_features[label] = {}
            for feature in df_features.columns:
                unique_values = subset[feature].unique()
                self.Sub_features[label][feature] = {}
                total_in_label = len(subset)
                for sub_feature in df_features[feature].unique():
                    count = len(subset[subset[feature] == sub_feature])
                    n = len(unique_values)
                    prob = (count + 1) / (total_in_label + n)
                    self.Sub_features[label][feature][sub_feature] = prob
        print(self.calculate_priors(self.Labels))
        result = self.predict(df_features.iloc[0].to_dict())
        print(result)

