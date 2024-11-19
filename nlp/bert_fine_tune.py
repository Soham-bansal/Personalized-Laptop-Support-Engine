
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def bert_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_bert_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # BERT Embeddings
    bert_embeddings = np.vstack([bert_embedding(desc) for desc in df["User Description"]])

    # Save BERT features
    np.save(output_path, bert_embeddings)
    print(f"BERT embeddings saved to {output_path}")

if __name__ == '__main__':
    input_path = '/mnt/data/preprocessed_dataset.csv'
    output_path = '/mnt/data/bert_features.npy'
    extract_bert_features(input_path, output_path)
