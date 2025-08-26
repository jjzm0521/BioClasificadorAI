import pandas as pd
import numpy as np
from data_preparation import preprocess_text, binarize_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split as standard_train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import joblib
import os
import time

# --- Configuration ---
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (src -> modelo-baseline)
project_root = os.path.dirname(script_dir)

DATA_FILE = os.path.join(project_root, 'data', 'raw', 'challenge_data-18-ago.csv')
PROCESSED_DATA_FILE = os.path.join(project_root, 'data', 'processed', 'processed_data.pkl')
MODEL_FILE = os.path.join(project_root, 'results', 'models', 'baseline_model.joblib')
TEST_DATA_FILE = os.path.join(project_root, 'results', 'models', 'test_data.joblib')

DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']
TEXT_COLUMN = 'text'
RANDOM_STATE = 42

# --- Main Training Script ---
def train_baseline():
    """
    Main function to load, preprocess, and train the baseline model.
    """
    # 1. Load and Preprocess Data
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Loading preprocessed data from {PROCESSED_DATA_FILE}...")
        df = pd.read_pickle(PROCESSED_DATA_FILE)
    else:
        print(f"Loading and preprocessing data from {DATA_FILE}...")
        try:
            df = pd.read_csv(DATA_FILE, sep=';', engine='python')
        except FileNotFoundError:
            print(f"Error: The file {DATA_FILE} was not found.")
            return

        # Binarize labels
        df = binarize_labels(df, DOMAINS)

        # Combine title and abstract
        df['abstract'] = df['abstract'].fillna('')
        df[TEXT_COLUMN] = df['title'] + ' ' + df['abstract']

        # Preprocess text (this is the slow part)
        print("Applying text preprocessing to all articles. This may take a few minutes...")
        start_time = time.time()
        df['processed_text'] = df[TEXT_COLUMN].apply(preprocess_text)
        end_time = time.time()
        print(f"Text preprocessing finished in {end_time - start_time:.2f} seconds.")

        # Save processed data to avoid repeating this step
        df.to_pickle(PROCESSED_DATA_FILE)
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")

    # 2. Split Data
    print("Splitting data into training, validation, and test sets...")

    X = df[['processed_text']]
    y = df[DOMAINS].values

    # skmultilearn requires numpy arrays
    X_np = np.array(X.index).reshape(-1, 1) # Use index to split, then retrieve text
    y_np = np.array(y)

    # First split: 70% train, 30% temp (val + test)
    X_train_idx, y_train, X_temp_idx, y_temp = iterative_train_test_split(X_np, y_np, test_size=0.3)

    # Second split: 15% val, 15% test (50% of the 30% temp)
    X_val_idx, y_val, X_test_idx, y_test = iterative_train_test_split(X_temp_idx, y_temp, test_size=0.5)

    # Retrieve text data using indices
    X_train = df.loc[X_train_idx.flatten(), 'processed_text']
    X_val = df.loc[X_val_idx.flatten(), 'processed_text']
    X_test = df.loc[X_test_idx.flatten(), 'processed_text']

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # 3. Define and Train Model Pipeline
    print("Defining and training the baseline model pipeline (TF-IDF + Logistic Regression)...")

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE), n_jobs=-1))
    ])

    # Train the model
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    print(f"Model training finished in {end_time - start_time:.2f} seconds.")

    # 4. Save the Model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    print(f"Saving the trained model to {MODEL_FILE}...")
    joblib.dump(pipeline, MODEL_FILE)

    # Also save the test set for evaluation
    os.makedirs(os.path.dirname(TEST_DATA_FILE), exist_ok=True)
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(test_data, TEST_DATA_FILE)
    print(f"Test data saved for evaluation to {TEST_DATA_FILE}")

    print("\n--- Training script finished successfully! ---")

if __name__ == '__main__':
    train_baseline()
