from data_loader import VoiceDataset
from model import ParkinsonPredictor
from utils import summarize_dataset, validate_status_value


DATA_PATH = "data/parkinsons.csv"
OUTPUT_DIR = "outputs"


def main():
    """Run the full project pipeline."""
    try:
        dataset = VoiceDataset(DATA_PATH)
        dataset.load()

        # Loop 1: show the first few feature names from the generator.
        print("First five features:")
        for index, feature in enumerate(dataset.feature_generator()):
            if index >= 5:
                break
            print("-", feature)

        # Loop 2: validate status values.
        for value in dataset.data["status"].unique():
            if not validate_status_value(value):
                raise ValueError("Invalid status value: " + str(value))

        summarize_dataset(dataset, OUTPUT_DIR)

        X_train, X_test, y_train, y_test = dataset.split()
        predictor = ParkinsonPredictor()
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)
        predictor.save_outputs(OUTPUT_DIR)
        predictor.save_predictions(X_test, y_test, OUTPUT_DIR)

        print(dataset)
        print(predictor)
        print("Confusion matrix:", metrics["confusion_matrix"])
        print("Results saved to: " + OUTPUT_DIR)

    except FileNotFoundError as error:
        print("File error:", error)
    except ValueError as error:
        print("Data error:", error)


if __name__ == "__main__":
    main()
