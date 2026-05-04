import json
import os
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class ParkinsonPredictor:
    """Base class for Parkinson's disease prediction models."""

    def __init__(self, random_state=42):
        """Initialize the predictor.
        
        Parameters:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.metrics = {}
        self.feature_importances = None
        self._trained = False

    def __str__(self):
        """Return a short description of the predictor."""
        if not self.metrics:
            return f"{self.__class__.__name__}: not trained"
        accuracy = round(self.metrics["accuracy"], 4)
        f1 = round(self.metrics["f1_score"], 4)
        return f"{self.__class__.__name__}: accuracy={accuracy}, f1={f1}"

    def __gt__(self, other):
        """Compare two predictors by F1 score."""
        self_f1 = self.metrics.get("f1_score", 0)
        other_f1 = other.metrics.get("f1_score", 0)
        return self_f1 > other_f1

    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics.
        
        Parameters:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing accuracy, f1_score, confusion_matrix, and classification_report
        """
        if not self._trained:
            raise AttributeError(f"{self.__class__.__name__} has not been trained yet. Call train() first.")
            
        predictions = self.predict(X_test)

        cm = confusion_matrix(y_test, predictions)
        cm_list = cm.tolist()

        self.metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
            "confusion_matrix": cm_list,
            "classification_report": classification_report(y_test, predictions, output_dict=True)
        }
        
        return self.metrics

    def save_outputs(self, output_dir, modelName):
        """Save the trained model and metrics to files.
        
        Parameters:
            output_dir (str): Directory path to save outputs
        """
        if not self._trained:
            raise AttributeError(f"{self.__class__.__name__} has not been trained yet. Call train() first.")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_filename = f"parkinsons_{self.__class__.__name__.lower()}_model.joblib"
        model_path = os.path.join(output_dir, model_filename)
        joblib.dump(self.classifier, model_path)

        run_info = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": self.__class__.__name__,
            "model": self._get_model_description(),
            "metrics": self.metrics,
        }

        metrics_path = os.path.join(output_dir, f"{modelName}_metrics.json")
        with open(metrics_path, "w") as file:
            json.dump(run_info, file, indent=4)

        # Save feature importances if available
        if self.feature_importances is not None:
            importances_path = os.path.join(output_dir, "feature_importances.joblib")
            joblib.dump(self.feature_importances, importances_path)

    def save_predictions(self, X_test, y_test, output_dir, modelName):
        """Save prediction results for the test set.
        
        Parameters:
            X_test: Test features
            y_test: Test labels
            output_dir (str): Directory path to save predictions
        """
        if not self._trained:
            raise AttributeError(f"{self.__class__.__name__} has not been trained yet. Call train() first.")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        predictions = self.predict(X_test)
        actual = list(y_test)
        results = pd.DataFrame({
            "actual_status": actual,
            "predicted_status": predictions,
        })
        output_path = os.path.join(output_dir, f"{modelName}_test_predictions.csv")
        results.to_csv(output_path, index=False)

    def _get_model_description(self):
        """Get model description. Override in subclasses."""
        return "Parkinson Predictor"


class SVMPredictor(ParkinsonPredictor):
    """Support Vector Machine predictor for Parkinson's disease prediction."""

    def __init__(self, random_state=42):
        """Create the scaler and SVM model."""
        super().__init__(random_state)
        self.scaler = StandardScaler()
        self.classifier = SVC(kernel="rbf", random_state=random_state)

    def train(self, X_train, y_train):
        """Train the SVM model.
        
        Parameters:
            X_train: Training features
            y_train: Training labels
        """
        X_train_scaled = self.scaler.fit(X_train).transform(X_train)
        self.classifier.fit(X_train_scaled, y_train)
        self._trained = True

    def predict(self, X):
        """Predict Parkinson's status for input rows.
        
        Parameters:
            X: Input features to predict on
            
        Returns:
            array: Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def _get_model_description(self):
        """Get SVM model description."""
        return "SVM with RBF kernel and StandardScaler"


class RandomForestPredictor(ParkinsonPredictor):
    """Random Forest predictor for Parkinson's disease prediction."""

    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        """Create the Random Forest model.
        
        Parameters:
            n_estimators (int): Number of trees in the forest. Default: 300
            max_depth (int): Maximum depth of the tree. Default: None (unlimited)
            random_state (int): Random state for reproducibility
        """
        super().__init__(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        )

    def train(self, X_train, y_train):
        """Train the Random Forest model.
        
        Parameters:
            X_train: Training features
            y_train: Training labels
        """
        self.classifier.fit(X_train, y_train)
        self._trained = True

    def predict(self, X):
        """Predict Parkinson's status for input rows.
        
        Parameters:
            X: Input features to predict on
            
        Returns:
            array: Predicted labels
        """
        return self.classifier.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the model and extract feature importances.
        
        Parameters:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing accuracy, f1_score, confusion_matrix, and classification_report
        """
        metrics = super().evaluate(X_test, y_test)
        self.feature_importances = self.classifier.feature_importances_
        return metrics

    def get_feature_importances(self, feature_cols=None):
        """Get feature importances from the Random Forest model.
        
        Parameters:
            feature_cols (list): List of feature column names
            
        Returns:
            pd.Series: Feature importances sorted in descending order
        """
        if self.feature_importances is None:
            return None
        
        if feature_cols is None:
            return pd.Series(self.feature_importances)
        
        importances = pd.Series(self.feature_importances, index=feature_cols)
        return importances.sort_values(ascending=False)

    def _get_model_description(self):
        """Get Random Forest model description."""
        return f"Random Forest (n_estimators={self.n_estimators}, class_weight=balanced)"