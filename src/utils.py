import math
from pathlib import Path


def validate_status_value(value):
    """Check whether a status value is valid (0 or 1)."""
    return value in (0, 1)


def summarize_dataset(dataset, output_dir):
    """Save a simple dataset summary to a text file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset.data is None:
        raise ValueError("Dataset is not loaded. Call dataset.load() first.")

    status_counts = dataset.data["status"].value_counts().to_dict()
    positive_rate = status_counts.get(1, 0) / len(dataset)

    # Use filter() and lambda to select frequency-related features.
    frequency_features = list(filter(lambda col: "Hz" in col, dataset.features))

    # Use the math module for a simple rounded percentage value.
    positive_percent = math.floor(positive_rate * 10000) / 100

    lines = [
        "Parkinson's Voice Dataset Summary",
        f"Rows: {len(dataset)}",
        f"Features used: {len(dataset.features)}",
        f"Status counts: {status_counts}",
        f"Parkinson's samples: {positive_percent}%",
        f"Frequency features: {frequency_features}",
    ]

    with open(output_path / "dataset_summary.txt", "w") as file:
        for line in lines:
            file.write(line + "\n")
