# Parkinson's Disease Detection Using Biomedical Voice Features

## Team Members
- Charmilkumar Vijaykumar Patel | SID: 20036845 | cpatel6@stevens.edu
- Yunyang Zhang | SID: 20043349 | yzhang102@stevens.edu

## Project Overview
This project uses biomedical voice measurements to detect whether a voice sample is related to Parkinson's disease. The project uses the Parkinson's dataset from the UCI Machine Learning Repository. The program loads the dataset, preprocesses the data, trains a machine learning model, evaluates model performance, and saves summary and visualization results.

## Project Objective
The main objective is to build a well-structured Python program that solves a real-world biomedical problem using object-oriented programming, data input/output, exception handling, Python libraries, and machine learning.

### Key Features
- **Data Processing:** Loads and preprocesses 195 voice samples from the Parkinson's dataset
- **Machine Learning:** Trains a classification model to distinguish healthy voices from Parkinson's-affected voices
- **Model Persistence:** Saves and loads trained models for reproducibility
- **Performance Analysis:** Generates metrics (accuracy, F1-score, confusion matrix) and visualizations
- **Modular Architecture:** Separates concerns across multiple Python modules and a Jupyter Notebook

### Dataset Information
- **Source:** Parkinson's Voice Dataset (publicly available, no IP issues)
- **Size:** 195 voice samples with 22 biomedical features
- **Target Variable:** `status` (0 = healthy, 1 = Parkinson's disease)
- **Features:** Voice-derived acoustic properties (jitter, shimmer, fundamental frequency, etc.)

## Dependencies
Install required libraries using:

```bash
pip install -r requirements.txt
```

Required libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
- pytest
- jupyter
- notebook

## File Structure

```text
AAI-551-WS/
│
├── data/
│   └── parkinsons.csv              # UCI Parkinson's voice dataset
│
├── results/                        # Generated outputs
│   ├── dataset_summary.txt
│   └── status_distribution.png
│
├── src/
│   ├── data_loader.py              # VoiceDataset class
│   ├── model.py                    # ParkinsonPredictor class
│   ├── utils.py                    # Helper functions
│   ├── visualization.py            # Plotting utilities
│
├── tests/
│   ├── __init__.py
│   ├── test1.py
│   ├── test2.py
│   └── test3.py
│
├── parkinsons_detection.ipynb      # Main Jupyter Notebook
├── requirements.txt
└── README.md
```

## How to Use

### Option A – Jupyter Notebook (recommended)
```bash
jupyter notebook parkinsons_detection.ipynb
```
## Model Results

### Random Forest Classifier
Accuracy:       91.26%

Precision:      92.11%

Recall:         96.88%

F1-Score:       94.45%

Confusion Matrix: [[4, 3], [1, 31]]

### Support Vector Machine (SVM)
Accuracy:       89.74%

Precision:      89.47%

Recall:         94.12%

F1-Score:       91.72%

Confusion Matrix: [[3, 4], [0, 32]]

## Requirements Coverage

### 🔹 Part 1 Requirements

| Requirement           | Details                                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Classes               | `VoiceDataset`, `ParkinsonPredictor` (base), `RandomForestPredictor`, `SVMPredictor` with constructors, attributes, and methods |
| Class Relationship    | Inheritance: `RandomForestPredictor` and `SVMPredictor` extend `ParkinsonPredictor`                                             |
| Functions             | `validate_status_value()`, `summarize_dataset()`, and additional preprocessing functions                                        |
| Advanced Libraries    | `pandas` (data manipulation), `scikit-learn` (Random Forest, SVM, metrics)                                                      |
| Exception Handling    | Handles missing CSV files, invalid data values, and runtime errors (2+ scenarios)                                               |
| Pytest Tests          | Unit tests in `tests/` directory (3+ test files, multiple test cases)                                                           |
| Data I/O              | Reads `parkinsons.csv`, writes JSON metrics and CSV predictions                                                                 |
| Loops & Conditionals  | Uses `for`, `while`, and `if/elif/else` statements                                                                              |
| Data Types            | Mutable: lists, dicts; Immutable: strings, tuples, numbers                                                                      |
| Operator Overloading  | `__str__()`, `__len__()`, `__eq__()`, `__gt__()`                                                                                |
| Docstrings & Comments | Comprehensive docstrings and inline comments                                                                                    |
| README                | Complete documentation with execution steps and setup instructions                                                              |

---

### 🔹 Part 2 Requirements

| Component                   | Location                                                |
| --------------------------- | ------------------------------------------------------- |
| `filter()`, `lambda`        | Used in data processing (`utils.py`)                    |
| Comprehensions              | List/dict comprehensions (`data_loader.py`, `utils.py`) |
| Built-in Libraries          | `json`, `pathlib`, `time`, `math`, `random`             |
| Generators                  | Generator function/expression (`utils.py`)              |
| Set Operations              | Feature validation and analysis (`model.py`)            |
| Recursion                   | Used for directory traversal or data processing         |

## Team Contributions

Charmilkumar Vijaykumar Patel: Set up the GitHub repository and managed version control, sourced and provided the Parkinson's voice dataset, and contributed to Python code implementation.

Yunyang Zhang: Contributed to Python code implementation, handled data preprocessing and program structure, and wrote the README and output documentation.

### Charmilkumar Vijaykumar Patel
- Set up the GitHub repository and managed version control
- Sourced and provided the Parkinson's voice dataset
- Implemented `ParkinsonPredictor` class and Random Forest Model
- Implemented `plot_pairplot` & `plot_feature_correlation` methods
- Wrote test cases using pytest.
- Created and maintained the Jupyter Notebook `parkinsons_detection.ipynb`.

### Yunyang Zhang
- Implemented `VoiceDataset` class, wrote code for data loading and processing.
- Implemented SVM model.
- Implemented `VisualizationManager` class and wrote `plot_label_distribution`, `plot_confusion_matrix`, `plot_feature_importance` methods.
- Wrote README with complete documentation.

