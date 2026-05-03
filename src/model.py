import json
import os
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ParkinsonPredictor:
    """Train and evaluate a Parkinson's disease prediction model."""

    def __init__(self, random_state=42):
        """Create the scaler and SVM model."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = SVC(kernel="rbf", random_state=random_state)
        self.metrics = {}

    def __str__(self):
        """Return a short description of the predictor."""
        if not self.metrics:
            return "ParkinsonPredictor: not trained"
        accuracy = round(self.metrics["accuracy"], 4)
        f1 = round(self.metrics["f1_score"], 4)
        return "ParkinsonPredictor: accuracy=" + str(accuracy) + ", f1=" + str(f1)

    def __gt__(self, other):
        """Compare two predictors by F1 score."""
        self_f1 = 0
        other_f1 = 0
        if "f1_score" in self.metrics:
            self_f1 = self.metrics["f1_score"]
        if "f1_score" in other.metrics:
            other_f1 = other.metrics["f1_score"]
        return self_f1 > other_f1

    def train(self, X_train, y_train):
        """Train the SVM model."""
        X_train_scaled = self.scaler.fit(X_train).transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.classifier.predict(X_test_scaled)

        cm = confusion_matrix(y_test, predictions)
        cm_list = cm.tolist()

        self.metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "confusion_matrix": cm_list,
        }
        return self.metrics

    def predict(self, X):
        """Predict Parkinson's status for input rows."""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def save_outputs(self, output_dir):
        """Save the trained model and metrics to files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_path = os.path.join(output_dir, "parkinsons_model.joblib")
        joblib.dump(self.classifier, model_path)

        run_info = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "SVM with StandardScaler",
            "metrics": self.metrics,
        }

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as file:
            json.dump(run_info, file, indent=4)

    def save_predictions(self, X_test, y_test, output_dir):
        """Save prediction results for the test set."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        predictions = self.predict(X_test)
        actual = list(y_test)
        results = pd.DataFrame({
            "actual_status": actual,
            "predicted_status": predictions,
        })
        output_path = os.path.join(output_dir, "test_predictions.csv")
        results.to_csv(output_path, index=False)
