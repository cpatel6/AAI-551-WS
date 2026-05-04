import math
import os
from scipy.stats import skew
import numpy as np
import pandas as pd

def validate_status_value(value):
    """Check whether a status value is valid (0 or 1)."""
    return value in (0, 1)

def validate_required_columns(df, required_columns):
    """Raise ValueError if any required columns are missing from the DataFrame."""
    available_columns = set(df.columns)
    required_set = set(required_columns)
    missing_columns = required_set - available_columns

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True


def summarize_dataset(dataset, output_dir):
    """Save a simple dataset summary to a text file."""
    if dataset.data is None:
        raise ValueError("Dataset is not loaded. Call dataset.load() first.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    summary = [
        "Parkinson's Voice Dataset Summary",
        "Rows: " + str(len(dataset)),
        "Missing Values: " + str(dataset.load().isnull().sum().sum()),
        "Features used: " + str(len(dataset.features)),
        "Status counts: " + str(status_counts),
        "Parkinson's samples: " + str(positive_percent) + "%",
        "Frequency features: " + str(frequency_features),
    ]

    output_path = os.path.join(output_dir, "dataset_summary.txt")
    with open(output_path, "w") as file:
        for line in summary:
            file.write(line + "\n")

    return summary

def summarize_vars(dataframe, columns, output, filename):
    rows = []
    for col in columns:
        s = dataframe[col].dropna()

        mean_val = s.mean()
        std_val = s.std(ddof=1)
        var_val = s.var(ddof=1)

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        rng = s.max() - s.min()

        skew_val = skew(s, bias=False)

        cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan

        rows.append({
            "Variable": col,
            "Count": int(s.size),
            "Mean": float(mean_val),
            "Std (sample)": float(std_val),
            "Variance (sample)": float(var_val),
            "CV (%)": float(cv) if not np.isnan(cv) else None,
            "Min": float(s.min()),
            "Q1": float(q1),
            "Median": float(s.median()),
            "Q3": float(q3),
            "Max": float(s.max()),
            "IQR": float(iqr),
            "Range": float(rng),
            "Skewness": float(skew_val),
        })

    cols_order = [
        "Variable", "Count", "Mean", "Std (sample)", "Variance (sample)", "CV (%)",
        "Min", "Q1", "Median", "Q3", "Max", "IQR", "Range", "Skewness"
    ]

    statSummary = pd.DataFrame(rows)[cols_order]
    try:
        statSummary.to_csv(f"{output / filename} .csv")
    except Exception as e:
        print(f'[EXCEPTION] File not created from Summary: {e}')
    return statSummary
