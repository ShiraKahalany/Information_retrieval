import os
import re
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

# Base directory and paths
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1'
output_dir = os.path.join(base_dir, 'doc2vec_matrices')  # תיקיה לשמירת הווקטורים
os.makedirs(output_dir, exist_ok=True)

# קובץ ה-Excel
excel_file = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1/posts_first_targil.xlsx'

# Group names (sheets in the Excel file)
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# Function to clean text (remove punctuation, numbers)
def clean_text(text):
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    
    # Tokenize text
    tokens = word_tokenize(text.lower())
    return tokens

# Process each group and create document vectors
for group in groups:
    print(f"Processing group: {group}")
    
    # Load the sheet from Excel
    sheet_data = pd.read_excel(excel_file, sheet_name=group)
    
    # Check if the necessary column exists
    if 'Body Text' not in sheet_data.columns:
        print(f"Skipping {group}: 'Body Text' column missing.")
        continue
    
    # Drop rows with missing text
    sheet_data = sheet_data.dropna(subset=['Body Text'])
    
    # Load and clean texts
    corpus = []
    for idx, text in enumerate(sheet_data['Body Text']):
        tokens = clean_text(str(text))
        corpus.append(TaggedDocument(tokens, [f"{group}_{idx}"]))
    
    # Train Doc2Vec model
    model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Create document vectors
    document_vectors = []
    for doc in corpus:
        vector = model.infer_vector(doc.words)  # Infer vector for the document
        document_vectors.append(vector)
    
    # Save the document vectors to a text file
    output_file = os.path.join(output_dir, f"{group}_doc2vec_vectors.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for vector in document_vectors:
            f.write("\t".join(map(str, vector)) + "\n")
    
    print(f"Saved vectors for group: {group}")

print("Doc2Vec document vectors have been saved.")
