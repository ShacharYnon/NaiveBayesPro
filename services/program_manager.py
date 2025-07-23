from services.data_loader import DataLoader
from model.classifier import Classifier
from model.debugger import Debugger
import os

class Task_manger:

    def __init__(self):
        self.df = None
        self.classifier = None
        self.features = None
        self.labels = None


    def process_data(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = "data"

        csv_path = os.path.join(project_root, base_dir, "buy_computer_data.csv")
        # csv_path = r"C:/Users/Work/OneDrive/Desktop/NaiveBayesProject/NaiveBayesPro/data/buy_computer_data.csv"
        self.df = DataLoader(csv_path, "buys_computer")
        # self.df = DataLoader(csv_path, "class")
        # self.df.drot_column("Index")
        self.df.read_csv()


    def train_and_predict (self):
        self.features, self.labels = self.df.get_features_and_labels()
        self.classifier = Classifier(self.features, self.labels)
        self.classifier.fit()
        Debugger.evaluate(self.classifier, self.features, self.labels)



    def predict_sample(self):
        if not self.classifier:
            print("Classifier is not trained. Please run train_and_predict first.")
            return

        sample = {
            "age": "<= 30",
            "income": "medium",
            "student": "yes",
            "credit_rating": "excellent"
        }

        pred = self.classifier.predict(sample)
        probs = self.classifier.predict_proba(sample)
        samples_df = self.features.head()
        batch_preds = self.classifier.predict_batch(samples_df)

        return {
            "Prediction:": pred,
            "Sample" : sample,
            "Probabilities:": probs
        }

        # return {
        #     "Prediction:" : pred,
        #     "Probabilities:" : probs,
        #     "Batch predictions:": batch_preds
        # }

if __name__ == "__main__":
    Task_manger()