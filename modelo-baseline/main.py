import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_model import train_baseline
from evaluate_model import evaluate_model

def main():
    """
    Main function to run the entire ML pipeline.
    1. Train the model.
    2. Evaluate the model.
    """
    print("--- Starting ML Pipeline ---")

    # Step 1: Train the model
    print("\n--- Running Training ---")
    train_baseline()

    # Step 2: Evaluate the model
    print("\n--- Running Evaluation ---")
    # Check if model and test data exist before evaluating
    model_path = os.path.join(os.path.dirname(__file__), 'results', 'models', 'baseline_model.joblib')
    test_data_path = os.path.join(os.path.dirname(__file__), 'results', 'models', 'test_data.joblib')
    if os.path.exists(model_path) and os.path.exists(test_data_path):
        evaluate_model()
    else:
        print("Evaluation skipped because model or test data was not found.")

    print("\n--- ML Pipeline Finished ---")

if __name__ == '__main__':
    main()
