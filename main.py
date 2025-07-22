from services.data_loader import DataLoader
from model.classifier2 import Classifier
from model.debugger import Debugger
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "buy_computer_data.csv")
    df = DataLoader(csv_path, "buys_computer")

    # csv_path = os.path.join(base_dir, "phishing.csv")
    # df = DataLoader(csv_path, "class")

    df.read_csv()
    # df.drot_column("Index")

    features, labels = df.get_features_and_labels()
    classifier = Classifier(features, labels)
    classifier.fit()

    Debugger.evaluate(classifier, features, labels)

    sample = {
    "age": "<=30",
    "income": "medium",
    "student": "yes",
    "credit_rating": "excellent"
    }
    pred = classifier.predict(sample)
    print("Prediction:", pred)

    probs = classifier.predict_proba(sample)
    print("Probabilities:", probs)

    samples_df = features.head()
    batch_preds = classifier.predict_batch(samples_df)
    print("Batch predictions:", batch_preds)


if __name__ =="__main__":
    main()




