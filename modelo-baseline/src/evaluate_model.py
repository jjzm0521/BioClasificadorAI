import joblib
from sklearn.metrics import f1_score, classification_report, hamming_loss
import pandas as pd
import os

# --- Configuration ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
TEST_DATA_FILE = os.path.join(project_root, 'results', 'models', 'test_data.joblib')
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Main Evaluation Script ---
def evaluate_model():
    """
    Loads a trained model and test data, evaluates the model,
    and prints the performance metrics.
    """
    # 1. Load Model and Data
    print(f"Loading model from {MODEL_FILE} and test data from {TEST_DATA_FILE}...")
    try:
        pipeline = joblib.load(MODEL_FILE)
        test_data = joblib.load(TEST_DATA_FILE)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please run the training script (train_model.py) first to generate the model and test data.")
        return

    # 2. Make Predictions
    print("Making predictions on the test set...")
    y_pred = pipeline.predict(X_test)

    # 3. Calculate and Print Metrics
    print("\n--- Model Evaluation Results ---")

    # Weighted F1-Score
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"\nWeighted F1-Score: {f1_weighted:.4f}")

    # Hamming Loss
    hamming = hamming_loss(y_test, y_pred)
    print(f"Hamming Loss: {hamming:.4f}")
    print("(Lower is better. It's the fraction of labels that are incorrectly predicted.)")

    # Classification Report (per-class metrics)
    print("\nClassification Report (per-class performance):")
    # Use target_names to label the classes in the report
    report = classification_report(y_test, y_pred, target_names=DOMAINS, zero_division=0)
    print(report)

    print("--- Evaluation script finished. ---")

if __name__ == '__main__':
    evaluate_model()
