import numpy as np

class Debugger:
    
    @staticmethod
    def debug_prediction(model, sample=None):
        if sample is None:
            sample = model.Features.iloc[0].to_dict()

        priors = model.calculate_priors()

        print("=== DEBUG PREDICTION ===")
        print(f"Sample: {sample}")
        print(f"Priors: {priors}")
        print(f"Labels available: {list(model.Sub_features.keys())}")

        for label in model.Sub_features:
            log_prob = np.log(priors[label])
            print(f"\nLabel {label}:")
            print(f"Prior log prob: {log_prob}")

            for feature, value in sample.items():
                feature_prob = model.Sub_features[label].get(feature, {}).get(value, 1e-6)
                log_prob += np.log(feature_prob)
                print(f"  {feature}={value}: prob={feature_prob}, cumulative_log={log_prob}")

            print(f"Final log prob for {label}: {log_prob}")

    @staticmethod
    def evaluate(model, x, y):
        if not model.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")
        correct = 0
        for i in range(len(x)):
            sample = x.iloc[i].to_dict()
            true_label = y.iloc[i]
            predicted = model.predict(sample)
            if predicted == true_label:
                correct += 1

        accuracy = correct / len(x)
        print(f"Accuracy: {accuracy:.2%}")
