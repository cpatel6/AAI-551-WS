# Parkinson's Disease Detection Using Biomedical Voice Features

Team members: Charmilkumar Vijaykumar Patel, Yunyang Zhang

## Project Overview

This project uses biomedical voice features to predict whether a voice sample is related to Parkinson's disease. The program reads the Parkinson's voice dataset from a CSV file, prepares the data, trains a machine learning model, evaluates the model, and saves the results.

The dataset contains 195 voice samples. The `name` column is used as a sample ID and is not used for training. The `status` column is the target label:

- `0`: healthy
- `1`: Parkinson's disease

## Files

```text
parkinsons_project_final/
├── data/
│   └── parkinsons.csv
├── outputs/
│   ├── dataset_summary.txt
│   ├── metrics.json
│   ├── parkinsons_model.joblib
│   └── test_predictions.csv
├── src/
│   ├── data_loader.py
│   ├── main.py
│   ├── model.py
│   └── utils.py
├── README.md
└── requirements.txt
```

## Dependencies

- `pandas` — data loading and preprocessing
- `scikit-learn` — machine learning model training and evaluation
- `joblib` — saving and loading the trained model
- `json` (built-in) — writing evaluation metrics to file
- `pathlib` (built-in) — handling file paths and output directories

## Requirements Covered

- Two classes: `VoiceDataset` and `ParkinsonPredictor`
- Class relationship: `main.py` uses `VoiceDataset` to prepare data and passes it to `ParkinsonPredictor`
- At least two functions: `validate_status_value()` and `summarize_dataset()`
- Advanced libraries: `pandas`, `scikit-learn`, and `joblib`
- Exception handling: missing CSV file and invalid/missing data values
- Data I/O: reads `parkinsons.csv` and writes output files
- Loops and if statements: used in `main.py` and helper functions
- Mutable data types: lists and dictionaries
- Immutable data types: strings, integers, floats, and tuples
- Operator overloads: `__str__()`, `__len__()`, and `__gt__()`
- Part 2 features: `filter()`, `lambda`, list comprehension, built-in modules (`json`, `pathlib`, `math`, `time`), generator function, set operations, and `if __name__ == "__main__"`
- Docstrings and comments are included in the Python files

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project from the main project folder:

```bash
python src/main.py
```

## Output

After running the program, the `outputs` folder will contain:

- `dataset_summary.txt`: basic dataset information
- `metrics.json`: model evaluation results
- `parkinsons_model.joblib`: saved trained model
- `test_predictions.csv`: actual and predicted test labels

## Model Result

Using an 80/20 train-test split with random state 42, the model result is:

```text
Accuracy: 0.9231
F1 score: 0.9508
Confusion matrix: [[7, 3], [0, 29]]
```

## Team Contributions

Charmilkumar Vijaykumar Patel: Set up the GitHub repository and managed version control, sourced and provided the Parkinson's voice dataset, and contributed to Python code implementation.

Yunyang Zhang: Contributed to Python code implementation, handled data preprocessing and program structure, and wrote the README documentation.
