
from server import *
from services import *
from model import *
from services.data_loader import DataLoader
from ui import *
from model.classifier import Classifier
from model.debugger import Debugger





# df = DataLoader(r"C:\Users\Work\Downloads\archive\phishing.csv", "class")
df = DataLoader(r"data\buy_computer_data.csv", "buys_computer")

df.read_csv()
# df.drot_column("Index")
features, labels = df.get_features_and_labels()
classifier = Classifier(features, labels)
classifier.fit()

# Debugger.debug_prediction(classifier)
Debugger.evaluate(classifier, features, labels)
# classifier.fit()
