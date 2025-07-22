import numpy as np

class Classifier:

    def __init__(self ,features ,labels):
        self.Features = features          # DataFrame של התכונות (features)
        self.Labels = labels              # סדרת התוויות (labels)
        self.Sub_features = {}            # מילון לשמירת הסתברויות מותנות לפי label ו-feature
        self.is_fitted = False            # מצב אימון המודל (אם עבר fit או לא)

    def calculate_priors(self):
        # מחשב הסתברויות בסיס P(label) עבור כל תווית
        total = len(self.Labels)
        priors = {}
        for label in self.Labels.unique():
            priors[label] = len(self.Labels[self.Labels == label]) / total
        return priors

    def predict(self, sample):
        # בודק האם המודל כבר עבר תהליך התאמה (fit)
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        # בדיקה אם קיימים מאפיינים שלא הופיעו באימון
        for feature in sample.keys():
            if feature not in self.Features.columns:
                raise ValueError(f"Unknown feature: {feature}")

        # בדיקה אם חסרים מאפיינים שנדרשים עבור התחזית
        for feature in self.Features.columns:
            if feature not in sample:
                raise ValueError(f"Missing feature: {feature}")

        best_label = None                 # תחזית התווית הכי סבירה
        best_log_prob = float("-inf")    # התחלת הסיכוי הלוגריתמי הכי נמוך

        # חישוב הסתברויות בסיס (prior) עבור כל תווית
        priors = self.calculate_priors()

        # מעבר על כל תווית אפשרית
        for label in self.Sub_features:
            # חישוב הלוגריתם של ההסתברות הבסיסית
            log_prob = np.log(priors[label])

            # מעבר על כל מאפיין ודירוג ההסתברות המשותפת
            for feature, value in sample.items():
                if feature in self.Sub_features[label] and value in self.Sub_features[label][feature]:
                    # הסתברות מותנית (P(feature=value | label))
                    prob = self.Sub_features[label][feature][value]
                else:
                    # Laplace smoothing במקרה שאין מידע מתאים (כדי להימנע מאפס)
                    class_size = sum(self.Labels == label)
                    k = len(self.Features[feature].unique())  # מספר ערכים אפשריים למאפיין
                    prob = 1 / (class_size + k)

                # חיבור ההסתברות הלוגריתמית הכוללת
                log_prob += np.log(prob)

            # בדיקה אם זו התחזית עם הסיכוי הגבוה ביותר עד כה
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_label = label

        return best_label

    def fit(self):
        # מגדיר שהמודל אומן
        self.is_fitted = True
        df_features = self.Features
        df_labels = self.Labels
        classes = df_labels.unique()   # כל התוויות הייחודיות בנתונים
        data = {}

        # מחשב את כל הערכים האפשריים לכל מאפיין
        feature_values = {
            feature: df_features[feature].unique()
            for feature in df_features.columns
        }

        # עבור כל תווית, עבור כל תכונה, מחשב הסתברויות מותנות
        for class_value in classes:
            class_indices = df_labels == class_value      # מסנן רק דוגמאות עם התווית הזו
            class_features = df_features[class_indices]

            data[class_value] = {}
            for feature in df_features.columns:
                data[class_value][feature] = {}
                values = feature_values[feature]
                k = len(values)    # מספר ערכים אפשריים לתכונה זו

                for value in values:
                    count = (class_features[feature] == value).sum()  # כמה דוגמאות עם ערך זה
                    total = len(class_features)                       # סה"כ דוגמאות לתווית זו
                    # חישוב הסתברות מותנית עם Laplace smoothing
                    prob = (count + 1) / (total + k)
                    data[class_value][feature][value] = prob

        # מחשב את הסתברויות הבסיס (priors)
        self.priors = self.calculate_priors()
        # שומר את הנתונים המותנים במודל
        self.Sub_features = data

    def predict_proba(self, sample):
        # בודק אם המודל כבר אומן
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        log_probs = {}

        # מחשב לוג הסתברויות עבור כל תווית
        for label in self.Sub_features:
            log_prob = np.log(self.priors[label])
            for feature, value in sample.items():
                if feature in self.Sub_features[label] and value in self.Sub_features[label][feature]:
                    prob = self.Sub_features[label][feature][value]
                else:
                    # Laplace smoothing במקרה של ערך חדש
                    class_size = sum(self.Labels == label)
                    k = len(self.Features[feature].unique())
                    prob = 1 / (class_size + k)
                log_prob += np.log(prob)
            log_probs[label] = log_prob

        # כדי למנוע underflow, מחסיר את הערך המקסימלי לפני ההסבר
        max_log = max(log_probs.values())
        exp_probs = {label: np.exp(log_prob - max_log)
                     for label, log_prob in log_probs.items()}
        total = sum(exp_probs.values())

        # מחזיר הסתברויות נורמליות (סכום ל-1)
        return {label: prob / total for label, prob in exp_probs.items()}

    def predict_batch(self, samples_df):
        # בודק אם המודל אומן
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        # מחזיר רשימה של תחזיות עבור כל שורה ב-DataFrame
        return [self.predict(row.to_dict())
                for _, row in samples_df.iterrows()]

