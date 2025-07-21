import numpy as np

class Classifier:

    def __init__(self ,features ,labels):
        self.Features = features
        self.Labels = labels
        self.Sub_features = {}
        self.is_fitted = False

    def calculate_priors(self):
        total = len(self.Labels)
        priors = {}
        for label in self.Labels.unique():
            priors[label] = len(self.Labels[self.Labels == label]) / total
        return priors

    def predict(self, sample):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        best_label = None
        best_log_prob = float("-inf")
        priors = self.calculate_priors()
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
        self.is_fitted = True
        df_features = self.Features
        df_labels = self.Labels
        classes = df_labels.unique()
        data ={}
        for class_value in classes:
            class_indices = df_labels == class_value
            class_features = df_features[class_indices]

            data[class_value] = {}
            for feature in df_features.columns:
                data[class_value][feature] = {}
                values = df_features[feature].unique()
                k = len(values)

                for value in values:
                    count = (class_features[feature] == value).sum()
                    total = len(class_features)
                    prob = (count + 1) / (total + k)
                    data[class_value][feature][value] = prob
        self.Sub_features = data
