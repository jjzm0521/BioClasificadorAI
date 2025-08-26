import joblib
import os
from data_preparation import preprocess_text

# --- Configuration ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Prediction Function ---
def predict_domains(text: str):
    """
    Loads the trained model and predicts the domains for a given text.

    Args:
        text (str): The input text to classify.

    Returns:
        dict: A dictionary with the predicted labels and their probabilities.
              Returns None if the model is not found.
    """
    # 1. Load Model
    try:
        pipeline = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_FILE}")
        print("Please run the training script (train_model.py) first.")
        return None

    # 2. Preprocess the input text
    processed_text = preprocess_text(text)

    # The pipeline expects an iterable (like a list or pandas Series)
    text_to_predict = [processed_text]

    # 3. Make Prediction
    # Use predict_proba to get probabilities for each class
    probabilities = pipeline.predict_proba(text_to_predict)

    # Use predict to get the binary predictions (0 or 1)
    predictions = pipeline.predict(text_to_predict)

    # 4. Format the output
    # The output of predict_proba is a list of arrays, one for each class
    # We take the first element since we only predict on one sample
    # The probabilities are for [class_0, class_1] for each label. We want the prob of class 1.

    # The output of predict is a 2D array, we take the first row
    predicted_labels = [DOMAINS[i] for i, prediction in enumerate(predictions[0]) if prediction == 1]

    # Create a dictionary of all domain probabilities
    domain_probabilities = {}
    for i, domain in enumerate(DOMAINS):
        # The second column [:, 1] is the probability of the positive class (1)
        domain_probabilities[domain] = probabilities[i][0][1]


    return {
        "predicted_labels": predicted_labels,
        "probabilities": domain_probabilities
    }

if __name__ == '__main__':
    # Example Usage
    sample_text = "The study investigated the effects of a new drug on cardiac rhythm in patients with heart failure."
    print(f"--- Predicting for sample text ---\n'{sample_text}'\n")

    # This requires the model to be trained first.
    # We add a check to see if the model exists before running the example.
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Please train the model by running 'python main.py' in the 'modelo-baseline' directory first.")
    else:
        results = predict_domains(sample_text)
        if results:
            print("--- Prediction Results ---")
            print(f"Predicted Labels: {results['predicted_labels']}")
            print("\nProbabilities per Domain:")
            for domain, prob in results['probabilities'].items():
                print(f"- {domain}: {prob:.4f}")
