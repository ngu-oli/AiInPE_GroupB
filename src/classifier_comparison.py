import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report


class ClassifierComparison:
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

        print(self.feature_columns)

        # Prepare the feature matrix (X) and target vector (y)
        self.X = self.data[self.feature_columns]
        self.y = self.data['Dataset_encoded']

        # filter features that have zero variance (feature selection function will issue a warning abut constant features)
        vt = VarianceThreshold(threshold=0)
        self.X = vt.fit_transform(self.X)

        # Standardize the features - this is mainly for the LogisticRegression Classifier. Seems to have no effect on the RandomForest Classifier.
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        # Get the remaining feature names
        selected_feature_indices = vt.get_support(indices=True)
        self.selected_features = [self.feature_columns[i] for i in selected_feature_indices]
        print("Selected features after VarianceThreshold:", self.selected_features)


    def print_class_mapping(self):
        # Print the mapping of encoded values to original class labels
        class_mapping = dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_))
        print("Class Mapping (Encoded Value to Original Label):")
        for k, v in class_mapping.items():
            print(f"{k}: {v}")


    def compare_classifiers(self, top_features=10):

        # Initial split into training+validation (80%) and testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # Select top k features based on ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=top_features)
        X_train_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)

        # Get the selected feature names
        selected_features_kbest = [self.selected_features[i] for i in selector.get_support(indices=True)]        
        print(f"Best features after feature selection: {selected_features_kbest}")


        # Define classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(solver='saga', max_iter=5000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
        }

       # save best classifier
        best_accuracy, best_report, best_name = 0, None, None

        # Train and evaluate classifiers
        results = {}
        print("-------------------------------------------------------------")
        print("Evaluating classifiers...\n")

        for name, clf in classifiers.items():
            # Train the classifier
            clf.fit(X_train_new, y_train)
            # Predict on the validation set
            y_pred = clf.predict(X_test_new)
            # Evaluate the classifier
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            results[name] = {"accuracy": accuracy, "classification_report": report}
            
            # Assign this classifier as best if better accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_report = report
                best_name = name

        # Display validation results
        for name, metrics in results.items():
            print(f"Classifier: {name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Classification Report:\n{metrics['classification_report']}\n")


        # print best classifier
        print("-------------------------------------------------------------")
        print(f"Best Accuracy: {best_accuracy} using the {best_name} classifier.")
        print(best_report)
