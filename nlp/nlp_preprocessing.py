
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Lemmatization function
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

# Load and preprocess dataset
def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df["User Description"] = df["User Description"].apply(lemmatize_text)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")

if __name__ == '__main__':
    input_path = '/mnt/data/reduced_complex_fake_neural_network_dataset_inr.csv'
    output_path = '/mnt/data/preprocessed_dataset.csv'
    preprocess_dataset(input_path, output_path)
