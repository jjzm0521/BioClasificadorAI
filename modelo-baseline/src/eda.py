import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re

# --- Configuration ---
DATA_FILE = r'modelo-baseline\data\raw\challenge_data-18-ago.csv'
OUTPUT_DIR = r'modelo-baseline\results\images'
DOMAINS = ['Cardiovascular', 'Neurological', 'Hepatorenal', 'Oncological']

# --- Helper Functions ---
def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text) # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

def plot_text_length_distribution(df, column_name, output_filename):
    """Plots and saves the distribution of text length for a given column."""
    plt.figure(figsize=(10, 6))
    df[f'{column_name}_length'] = df[column_name].str.len()
    sns.histplot(df[f'{column_name}_length'], bins=50, kde=True)
    plt.title(f'Distribution of {column_name.capitalize()} Length')
    plt.xlabel('Length (number of characters)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()
    print(f"Saved {column_name} length distribution plot to {output_filename}")

def generate_word_cloud(text, output_filename):
    """Generates and saves a word cloud from a block of text."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.close()
    print(f"Saved word cloud to {output_filename}")

# --- Main EDA Script ---
def perform_eda():
    """Main function to run the Exploratory Data Analysis."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Data
    print(f"Loading data from {DATA_FILE}...")
    try:
        # Corrected the parser by adding the separator and engine
        df = pd.read_csv(DATA_FILE, sep=';', engine='python', on_bad_lines='warn')
    except FileNotFoundError:
        print(f"Error: The file {DATA_FILE} was not found in the root directory.")
        return

    print("\n--- Initial Data Inspection ---")
    print("Data Head:")
    print(df.head())
    print("\nData Info:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Fill missing abstracts with empty string
    df['abstract'] = df['abstract'].fillna('')

    # 2. Analyze Labels (Domains)
    print("\n--- Domain/Label Analysis ---")
    # Binarize labels
    for domain in DOMAINS:
        df[domain] = df['group'].apply(lambda x: 1 if domain.lower() in x.lower() else 0)

    label_counts = df[DOMAINS].sum().sort_values(ascending=False)
    print("\nArticle counts per single domain:")
    print(label_counts)

    # Plot label distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Article Count per Medical Domain')
    plt.ylabel('Number of Articles')
    plt.xlabel('Domain')
    plt.savefig(os.path.join(OUTPUT_DIR, 'domain_distribution.png'))
    plt.close()
    print("Saved domain distribution plot to domain_distribution.png")

    # Analyze label combinations
    df['domain_combination'] = df[DOMAINS].apply(lambda row: '|'.join(row.index[row == 1]), axis=1)
    combo_counts = df['domain_combination'].value_counts()
    print("\nArticle counts per domain combination:")
    print(combo_counts)

    # 3. Analyze Text Length
    print("\n--- Text Length Analysis ---")
    df['text'] = df['title'] + ' ' + df['abstract']
    plot_text_length_distribution(df, 'title', 'title_length_dist.png')
    plot_text_length_distribution(df, 'abstract', 'abstract_length_dist.png')
    plot_text_length_distribution(df, 'text', 'full_text_length_dist.png')


    # 4. Generate Word Clouds
    print("\n--- Word Cloud Generation ---")
    # Clean text for word clouds
    df['cleaned_text'] = df['text'].apply(clean_text)

    for domain in DOMAINS:
        print(f"Generating word cloud for {domain}...")
        # Concatenate all text for the given domain
        domain_text = " ".join(df[df[domain] == 1]['cleaned_text'])
        if domain_text:
            generate_word_cloud(domain_text, f'wordcloud_{domain.lower()}.png')
        else:
            print(f"No text available to generate word cloud for {domain}")

    print("\nEDA script finished successfully.")

if __name__ == '__main__':
    perform_eda()
