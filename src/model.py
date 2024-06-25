import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


class FeatureImportanceEvaluator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.feature_importance_df = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        self.data = pd.read_excel(self.filepath)

    def preprocess_data(self):
        # Encode the target variable
        self.data['Dataset_encoded'] = self.label_encoder.fit_transform(self.data['Dataset'])

        # Select the feature columns (excluding 'Number of Measurement', 'Dataset', and 'Dataset_encoded')
        self.feature_columns = [col for col in self.data.columns if
                                col not in ['Number of Measurement', 'Dataset', 'Dataset_encoded']]

        # Prepare the feature matrix (X) and target vector (y)
        self.X = self.data[self.feature_columns]
        self.y = self.data['Dataset_encoded']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def train_model(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(self.X_train, self.y_train)

    def evaluate_feature_importance(self):

        feature_importances = self.rf_classifier.feature_importances_
        self.feature_importance_df = pd.DataFrame({'Feature': self.feature_columns, 'Importance': feature_importances})
        self.feature_importance_df = self.feature_importance_df.sort_values(by='Importance', ascending=False)

    def plot_feature_importance(self, top_n=10):
        if self.feature_importance_df is not None:
            top_features = self.feature_importance_df.head(top_n)
            plt.figure(figsize=(12, 8))
            plt.barh(np.array(top_features['Feature'])[::-1], np.array(top_features['Importance'])[::-1])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Top Feature Importance for Classification')
            plt.show()
        else:
            print("Feature importance has not been evaluated. Please run evaluate_feature_importance() first.")

    def get_top_features(self, top_n=10):
        if self.feature_importance_df is not None:
            return self.feature_importance_df.head(top_n)
        else:
            print("Feature importance has not been evaluated. Please run evaluate_feature_importance() first.")
            return None

    def train_and_evaluate_top_features_model(self, top_n=10):
        if self.feature_importance_df is not None:
            top_features = self.get_top_features(top_n)
            top_feature_columns = top_features['Feature'].tolist()

            # Prepare the feature matrix (X) with top features
            X_top = self.data[top_feature_columns]
            y = self.data['Dataset_encoded']

            # Split the data into training and testing sets
            X_train_top, X_test_top, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

            # Train the model with top features
            rf_classifier_top = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier_top.fit(X_train_top, y_train)

            # Evaluate the model
            y_pred = rf_classifier_top.predict(X_test_top)
            report = classification_report(y_test, y_pred)
            print(report)
        else:
            print("Feature importance has not been evaluated. Please run evaluate_feature_importance() first.")

    def print_class_mapping(self):
        # Print the mapping of encoded values to original class labels
        class_mapping = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        print("Class Mapping (Encoded Value to Original Label):")
        for k, v in class_mapping.items():
            print(f"{k}: {v}")


# Hier den Path angeben an dem die Daten (MIT FEATURE) gespeichert sind!
filepath = r'C:\Users\CT-Laptop-01\PycharmProjects\AiInPE_GroupB\src\Data\0001_Database_with_features.xlsx'
feature_importance_evaluator = FeatureImportanceEvaluator(filepath)
feature_importance_evaluator.load_data()
feature_importance_evaluator.preprocess_data()
feature_importance_evaluator.train_model()
feature_importance_evaluator.evaluate_feature_importance()

# Plot the top 10 features
feature_importance_evaluator.plot_feature_importance()

# Get the top 10 features
top_features = feature_importance_evaluator.get_top_features()
print(top_features)

feature_importance_evaluator.train_and_evaluate_top_features_model(top_n=6)
feature_importance_evaluator.print_class_mapping()