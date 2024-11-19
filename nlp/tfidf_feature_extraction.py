
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Feature Extraction
def extract_tfidf_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df["User Description"])

    # Save TF-IDF features
    np.save(output_path, tfidf_matrix.toarray())
    print(f"TF-IDF features saved to {output_path}")

if __name__ == '__main__':
    input_path = '/mnt/data/preprocessed_dataset.csv'
    output_path = '/mnt/data/tfidf_features.npy'
    extract_tfidf_features(input_path, output_path)
