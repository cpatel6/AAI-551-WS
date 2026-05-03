import math
import os


def validate_status_value(value):
    """Check whether a status value is valid (0 or 1)."""
    return value in (0, 1)


def summarize_dataset(dataset, output_dir):
    """Save a simple dataset summary to a text file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset.data is None:
        raise ValueError("Dataset is not loaded. Call dataset.load() first.")

    # Count how many samples are healthy (0) and Parkinson's (1).
    status_series = dataset.data["status"].value_counts()
    status_counts = status_series.to_dict()

    # Get the count of Parkinson's samples.
    if 1 in status_counts:
        positive_count = status_counts[1]
    else:
        positive_count = 0

    positive_rate = positive_count / len(dataset)

    # Use filter() and lambda to select frequency-related features.
    frequency_features = list(filter(lambda col: "Hz" in col, dataset.features))

    # Use the math module for a simple rounded percentage value.
    positive_percent = math.floor(positive_rate * 10000) / 100

    lines = [
        "Parkinson's Voice Dataset Summary",
        "Rows: " + str(len(dataset)),
        "Features used: " + str(len(dataset.features)),
        "Status counts: " + str(status_counts),
        "Parkinson's samples: " + str(positive_percent) + "%",
        "Frequency features: " + str(frequency_features),
    ]

    output_path = os.path.join(output_dir, "dataset_summary.txt")
    with open(output_path, "w") as file:
        for line in lines:
            file.write(line + "\n")
