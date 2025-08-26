import spacy
import re
import pandas as pd

# It's good practice to load the model once and pass it to functions.
# We will load it here and handle potential errors.
try:
    NLP = spacy.load("en_core_sci_lg", disable=["parser", "ner"])
    NLP.max_length = 2000000 # Increase max length for long abstracts
except OSError:
    print("Error: Model 'en_core_sci_lg' not found.")
    print("Please run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz")
    NLP = None

def preprocess_text(text: str) -> str:
    """
    Cleans, tokenizes, and lemmatizes text using the scispaCy NLP model.
    Removes stop words and non-alphabetic tokens.

    Args:
        text (str): The input string to process.

    Returns:
        str: The processed text with lemmas joined by spaces.
    """
    if NLP is None or not isinstance(text, str):
        return ""

    # Basic cleaning
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Process with scispaCy
    doc = NLP(text)

    # Lemmatize and remove stop words and non-alpha tokens
    lemmas = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

    return " ".join(lemmas)

def binarize_labels(df: pd.DataFrame, domains: list) -> pd.DataFrame:
    """
    Converts the 'group' column containing label strings into binary columns.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'group' column.
        domains (list): A list of the target domain names.

    Returns:
        pd.DataFrame: The DataFrame with new binary columns for each domain.
    """
    df_copy = df.copy()
    for domain in domains:
        df_copy[domain] = df_copy['group'].apply(
            lambda x: 1 if isinstance(x, str) and domain.lower() in x.lower() else 0
        )
    return df_copy

if __name__ == '__main__':
    # Example Usage
    print("Testing preprocessing functions...")

    # Test text preprocessing
    sample_text = "Effects of suprofen on the isolated perfused rat kidney. Although suprofen has been associated with the development of acute renal failure."
    print(f"Original:  {sample_text}")
    processed_text = preprocess_text(sample_text)
    print(f"Processed: {processed_text}")

    # Test label binarization
    sample_data = {
        'title': ['t1', 't2', 't3', 't4'],
        'abstract': ['a1', 'a2', 'a3', 'a4'],
        'group': ['cardiovascular', 'neurological|hepatorenal', 'oncological', 'cardiovascular|oncological']
    }
    sample_df = pd.DataFrame(sample_data)
    domains_list = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

    print("\nOriginal DataFrame:")
    print(sample_df)

    binarized_df = binarize_labels(sample_df, domains_list)
    print("\nBinarized DataFrame:")
    print(binarized_df)

    # Verify the output columns
    print(f"\nColumns in new DataFrame: {binarized_df.columns.tolist()}")
    print("\nPreprocessing module tests complete.")
