import numpy as np

class Predictor:

    def __init__(self ,model):
        self.model = model

    def predict(self, sample):
        best_label = None
        best_log_prob = float("-inf")
        priors = self.model.calculate_priors()

        for label in self.model.Sub_features:
            log_prob = np.log(priors[label])
            for feature, value in sample.items():
                prob = self.model.Sub_features[label].get(feature, {}).get(value, 1e-6)
                log_prob += np.log(prob)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_label = label
        return best_label