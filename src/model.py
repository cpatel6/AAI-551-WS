import json
import time
from pathlib import Path

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
        return f"ParkinsonPredictor: accuracy={self.metrics['accuracy']:.4f}, f1={self.metrics['f1_score']:.4f}"

    def __gt__(self, other):
        """Compare two predictors by F1 score."""
        return self.metrics.get("f1_score", 0) > other.metrics.get("f1_score", 0)

    def train(self, X_train, y_train):
        """Train the SVM model."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.classifier.predict(X_test_scaled)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
        return self.metrics

    def predict(self, X):
        """Predict Parkinson's status for input rows."""
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def save_outputs(self, output_dir):
        """Save the trained model and metrics to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.classifier, output_path / "parkinsons_model.joblib")

        run_info = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "SVM with StandardScaler",
            "metrics": self.metrics,
        }

        with open(output_path / "metrics.json", "w") as file:
            json.dump(run_info, file, indent=4)

    def save_predictions(self, X_test, y_test, output_dir):
        """Save prediction results for the test set."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions = self.predict(X_test)
        results = pd.DataFrame({
            "actual_status": y_test.values,
            "predicted_status": predictions,
        })
        results.to_csv(output_path / "test_predictions.csv", index=False)
