import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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
        self.feature_columns = [col for col in self.data.columns if col not in ['Number of Measurement', 'Dataset', 'Dataset_encoded']]

        # Prepare the feature matrix (X) and target vector (y)
        self.X = self.data[self.feature_columns]
        self.y = self.data['Dataset_encoded']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(self.X_train, self.y_train)

    def evaluate_feature_importance(self):
        feature_importances = self.rf_classifier.feature_importances_
        self.feature_importance_df = pd.DataFrame({'Feature': self.feature_columns, 'Importance': feature_importances})
        self.feature_importance_df = self.feature_importance_df.sort_values(by='Importance', ascending=False)

    def plot_feature_importance_random_forest(self, top_n=25):
        if self.feature_importance_df is not None:
            top_features = self.feature_importance_df.head(top_n)
            plt.figure(figsize=(12, 8))
            plt.barh(np.array(top_features['Feature'])[::-1], np.array(top_features['Importance'])[::-1])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Top Feature Importance for Classification (Random Forest)')
            plt.show()
        else:
            print("Feature importance has not been evaluated. Please run evaluate_feature_importance() first.")

    def get_top_features_random_forest(self, top_n=10):
        if self.feature_importance_df is not None:
            return self.feature_importance_df.head(top_n)
        else:
            print("Feature importance has not been evaluated. Please run evaluate_feature_importance() first.")
            return None
        
    def get_top_features_selectkbest(self, k=10):
        # Select top k features based on ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(self.X, self.y)
        selected_indices = selector.get_support(indices=True)
        k_best_features = [self.feature_columns[i] for i in selected_indices]
        return k_best_features
    
    def plot_feature_importance_selectkbest(self, k=10):
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(self.X, self.y)
        scores = selector.scores_[selector.get_support()]
        selected_indices = selector.get_support(indices=True)
        k_best_features = [self.feature_columns[i] for i in selected_indices]

        # Sort the features and scores in descending order of scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_features = np.array(k_best_features)[sorted_indices]
        sorted_scores = scores[sorted_indices]

        plt.figure(figsize=(12, 8))
        plt.barh(sorted_features[::-1], sorted_scores[::-1])
        plt.xlabel('Feature Importance (ANOVA F-value)')
        plt.ylabel('Feature')
        plt.title(f'Top {k} Feature Importance for Classification (SelectKBest)')
        plt.show()

    def train_and_evaluate_top_features_model(self, top_n=10):
        if self.feature_importance_df is not None:
            # Use the features selected by the RandomForest.
            top_features = self.get_top_features_random_forest(top_n)
            top_feature_columns = top_features['Feature'].tolist()

            # Uncomment this to use the features selected by SelectKBest.
            # top_features = set(self.get_top_features_selectkbest(k=top_n))
            # top_feature_columns = list(top_features)

            print("Selected Features: ")
            print(top_feature_columns)

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
            report = classification_report(y_test, y_pred, digits=4)
            print(report)
        else:
            print("Feature importance has not been evaluated. Please run evaluate_feature_importance() first.")

    def cross_validate_model(self, cv=5):
        # Perform cross-validation
        scores = cross_val_score(self.rf_classifier, self.X, self.y, cv=cv)
        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean Score: {np.mean(scores)}")
        print(f"Standard Deviation: {np.std(scores)}")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def print_class_mapping(self):
        # Print the mapping of encoded values to original class labels
        class_mapping = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        print("Class Mapping (Encoded Value to Original Label):")
        for k, v in class_mapping.items():
            print(f"{k}: {v}")
