import os

import pandas as pd
from sklearn.model_selection import train_test_split


class VoiceDataset:
    """Load and prepare the Parkinson's voice dataset."""

    def __init__(self, csv_path):
        """Create a VoiceDataset object."""
        self.csv_path = csv_path
        self.data = None
        self.features = []
        self.target = "status"

    def __str__(self):
        """Return a short description of the dataset."""
        if self.data is None:
            return "VoiceDataset: data not loaded"
        return "VoiceDataset: " + str(len(self)) + " rows, " + str(len(self.features)) + " features"

    def __len__(self):
        """Return the number of rows in the dataset."""
        if self.data is None:
            return 0
        return len(self.data)

    def load(self):
        """Load the CSV file and validate the required columns."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError("CSV file not found: " + self.csv_path)

        self.data = pd.read_csv(self.csv_path)

        # Use set operations to check for missing required columns.
        required_columns = {"name", "status"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError("Missing required columns: " + str(missing_columns))

        # Use list comprehension to keep all biomedical voice columns only.
        self.features = [col for col in self.data.columns if col not in ["name", "status"]]
        return self.data

    def split(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        if self.data is None:
            self.load()

        X = self.data[self.features]
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def feature_generator(self):
        """Yield feature names one at a time."""
        for feature in self.features:
            yield feature
