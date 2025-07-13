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




    # def fit(self):
        # df_features = self.Features
        # df_labels = self.Labels
        # for label in df_labels.unique():
        #     subset = df_features[df_labels == label]
        #     self.Sub_features[label] = {}
        #     for feature in df_features.columns:
        #         unique_values = subset[feature].unique()
        #         self.Sub_features[label][feature] = {}
        #         total_in_label = len(subset)
        #         for sub_feature in df_features[feature].unique():
        #             count = len(subset[subset[feature] == sub_feature])
        #             n = len(unique_values)
        #             prob = (count + 1) / (total_in_label + n)
        #             self.Sub_features[label][feature][sub_feature] = prob
        # print(self.calculate_priors(self.Labels))
        # result = self.predict(df_features.iloc[0].to_dict())
        # print(result)






    def table_to_dictionary(self):
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
        print(self.calculate_priors(self.Labels))
        result = self.predict(df_features.iloc[0].to_dict())
        print(result)

        def debug_prediction(self):
            sample = self.Features.iloc[0].to_dict()
            priors = self.calculate_priors(self.Labels)

            print("=== DEBUG PREDICTION ===")
            print(f"Sample: {sample}")
            print(f"Priors: {priors}")
            print(f"Labels available: {list(self.Sub_features.keys())}")

            for label in self.Sub_features:
                log_prob = np.log(priors[label])
                print(f"\nLabel {label}:")
                print(f"Prior log prob: {log_prob}")

                for feature, value in sample.items():
                    feature_prob = self.Sub_features[label].get(feature, {}).get(value, 1e-6)
                    log_prob += np.log(feature_prob)
                    print(f"  {feature}={value}: prob={feature_prob}, cumulative_log={log_prob}")

                print(f"Final log prob for {label}: {log_prob}")

        # import pprint
        # pprint.pprint(self.Sub_features)
        # print(self.Sub_features)





    #             self.Sub_features[feature] = df_features[feature].unique()
    #             # data[][][]
    #
    #
    #     count_true = len([lable for lable in self.Labels if lable == 'yes'])
    #     print(count_true)
    #     total = len([lable for lable in self.Labels])
    #     prior_x = count_true / total
    #     print(f"total: {total} \ncount_true: {count_true} \nprior_x: {prior_x}")
    #
    #
    #
    #
    #
    #
    #
    # #     data = {0:{}, 1: {}}
    # #     df_features = self.Features
    # #     df_labels = self.Labels
    # #     for column in df_features.column():
    # #         self.Sub_features[column] = df_features[column].unique()
    #
    # #     count_true = len([ lable for lable in self.Labels if lable == 'yes'])
    # #     total = len([lable for lable in self.Labels])
    # #     prior_x = count_true / total
    # #     print(f'prior_x - {prior_x}')
    # #     prior_y = 1 - prior_x
    # #     print(f'prior_y - {prior_y}')
    # #     for label in range(0, 2):
    # #         for feature in self.Features:
    # #             for sub_feature in self.Sub_features[feature]:
    # #                 data[label][feature][sub_feature] = prior_x * self.Sub_features[feature][sub_feature]
    # #     print(data)
    # #
    # # def get_probability_from_features(self):
    # #
    # #     for feature in self.Features:
    # #         age_counts = self.Features[feature].value_counts() / self.Features[feature].count()
    # #         print(age_counts)
    # #
    # #
    # #     # age_counts =  self.Features['age'].value_counts()/ self.Features['age'].count()
    # #     # print(age_counts)
    # #

#
# loader = DataLoader(r"C:\Users\Work\Downloads\archive\phishing.csv", "class")
# loader.read_csv()
# features, lables = loader.get_features_and_labels()
# controler = Classifier(features, lables)
# controler.table_to_dictionary()
# # controler.get_probability_from_features()
# # controler.table_to_dictionary()

# loader = DataLoader(r"C:\Users\Work\Downloads\archive\phishing.csv", "class")
# loader.read_csv()
# features, lables = loader.get_features_and_labels()
# controler = Classifier(features, lables)
# controler.table_to_dictionary()
#
# # הוסף את השורות הבאות:
# print("=== בדיקת הנתונים ===")
# print("שורה ראשונה:", features.iloc[0].to_dict())
# print("התווית האמיתית:", lables.iloc[0])
# print("כל התוויות במדגם:", lables.unique())
# print("ספירת כל תווית:", lables.value_counts())
# print("כל התוויות ב-Sub_features:", list(controler.Sub_features.keys()))
#
# # בדיקת החיזוי
# sample = features.iloc[0].to_dict()
# priors = controler.calculate_priors(lables)
# print("\nPriors:", priors)
#
# print("\n=== בדיקת החיזוי ===")
# result = controler.predict(sample)
# print("תוצאת החיזוי:", result)
#
#
# loader = DataLoader(r"C:\Users\Work\Downloads\archive\phishing.csv", "class")
# loader.read_csv()
# features, lables = loader.get_features_and_labels()
#
# # הוסף את השורה הזו:
# features = features.drop(columns=['Index'])
#
# controler = Classifier(features, lables)
# controler.table_to_dictionary()
#
# # עכשיו תבדוק:
# result = controler.predict(features.iloc[0].to_dict())
# print("תוצאה:", result)
# print("אמור להיות:", lables.iloc[0])



loader = DataLoader(r"C:\Users\Work\Downloads\archive\phishing.csv", "class")
loader.read_csv()
features, lables = loader.get_features_and_labels()

# הסר את האינדקס
features = features.drop(columns=['Index'])

controler = Classifier(features, lables)
controler.table_to_dictionary()

# בדיקה מפורטת
print("=== בדיקה ללא אינדקס ===")
print("שורה ראשונה (ללא Index):", features.iloc[0].to_dict())
print("התווית האמיתית:", lables.iloc[0])

# בדיקת החיזוי עם debug
sample = features.iloc[0].to_dict()
priors = controler.calculate_priors(lables)

print("\n=== בדיקת החיזוי המפורטת ===")
best_label = None
best_log_prob = float("-inf")

for label in controler.Sub_features:
    log_prob = np.log(priors[label])
    print(f"\nLabel {label}:")
    print(f"Prior log prob: {log_prob:.4f}")

    for feature, value in sample.items():
        feature_prob = controler.Sub_features[label].get(feature, {}).get(value, 1e-6)
        log_prob += np.log(feature_prob)
        if feature in ['UsingIP', 'LongURL', 'HTTPS']:  # רק כמה תכונות לדוגמה
            print(f"  {feature}={value}: prob={feature_prob:.6f}")

    print(f"Final log prob for {label}: {log_prob:.4f}")

    if log_prob > best_log_prob:
        best_log_prob = log_prob
        best_label = label

print(f"\nBest label: {best_label}")
print(f"צריך להיות: {lables.iloc[0]}")