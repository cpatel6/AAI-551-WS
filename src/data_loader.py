"""Data loading tools for the Parkinson's voice project."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class VoiceDataset:
    """Load and prepare the Parkinson's voice dataset.

    The class reads the CSV file, checks required columns, removes the sample
    name column, and prepares train/test data for the prediction model.
    """

    def __init__(self, csv_path):
        """Create a VoiceDataset object.

        Args:
            csv_path (str): Path to the Parkinson's CSV file.
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.features = []
        self.target = "status"

    def __str__(self):
        """Return a short description of the dataset."""
        if self.data is None:
            return "VoiceDataset: data not loaded"
        return f"VoiceDataset: {len(self)} rows, {len(self.features)} features"

    def __len__(self):
        """Return the number of rows in the dataset."""
        if self.data is None:
            return 0
        return len(self.data)

    def load(self):
        """Load the CSV file and validate the required columns.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is missing required columns.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path)
        required_columns = {"name", "status"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Use comprehension to keep all biomedical voice columns only.
        self.features = [col for col in self.data.columns if col not in ["name", "status"]]
        return self.data

    def split(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets.

        Args:
            test_size (float): Fraction of rows used for testing.
            random_state (int): Random seed for reproducible results.

        Returns:
            tuple: X_train, X_test, y_train, y_test.
        """
        if self.data is None:
            self.load()

        X = self.data[self.features]
        y = self.data[self.target]

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    def feature_generator(self):
        """Yield feature names one at a time.

        Yields:
            str: One feature name per iteration.
        """
        for feature in self.features:
            yield feature
