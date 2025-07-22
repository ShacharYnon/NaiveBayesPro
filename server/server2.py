# from fastapi import FastAPI
# from pydantic import BaseModel
# from model.classifier2 import Classifier
# from services.data_loader import DataLoader
#
# app = FastAPI()
#
# # טען את המודל פעם אחת
# df = DataLoader(r"data\buy_computer_data.csv", "buys_computer")
# df.read_csv()
# features, labels = df.get_features_and_labels()
# clf = Classifier(features, labels)
# clf.fit()  # תכניס פה את הדאטה שלך
#
# sample = {
#     "age": "<=30",
#     "income": "medium",
#     "student": "yes",
#     "credit_rating": "excellent"
# }
#
# result = clf.predict(sample)
# print(result)
#
# class InputData(BaseModel):
#     age: str
#     income: str
#     student: str
#     credit_rating: str
#
# @app.post("/predict")
# def predict(data: InputData):
#     sample = data.dict()
#     prediction = clf.predict(sample)
#     return {"prediction": prediction}
#
