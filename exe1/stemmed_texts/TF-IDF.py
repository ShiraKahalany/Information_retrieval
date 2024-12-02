import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import json

# Create directories to store the result
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/exe1'
clean_dir = os.path.join(base_dir, 'clean_texts')
stem_dir = os.path.join(base_dir, 'stemmed_texts')

# Group names
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# Prepare the output file to store the vectors
output_file = os.path.join(base_dir, 'document_vectors.json')

# Function to get all file paths in a directory for each group
def get_file_paths(group):
    group_dir = os.path.join(clean_dir, group)
    return [os.path.join(group_dir, file) for file in os.listdir(group_dir) if file.endswith('_clean.txt')]

# Load the cleaned text from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Apply TF-IDF and BM25/Okapi (heuristic adjustment)
def compute_tfidf_bm25(corpus):
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, norm='l2')
    
    # Ensure the corpus is a list of documents (2D)
    tfidf_matrix = vectorizer.fit_transform(corpus)  # This will return a 2D sparse matrix
    
    # To simulate BM25, adjust the IDF with a heuristic factor (BM25 formula: k1 / (1 + tf))
    idf = vectorizer.idf_
    
    # BM25 adjustment (we apply it only to the term frequencies, not the whole matrix)
    k1 = 1.5  # BM25 heuristic parameter (can be adjusted)
    
    # Element-wise scaling of the term frequencies in the sparse matrix (keep it sparse)
    tf_matrix = tfidf_matrix.copy()
    tf_matrix.data = k1 / (1 + tf_matrix.data)
    
    # Multiply term frequencies with IDF values
    bm25_matrix = tf_matrix.multiply(idf)
    
    return bm25_matrix, vectorizer

# Apply dimensionality reduction using SVD
def reduce_dimensions(matrix, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(matrix)
    return reduced_matrix

# Process each group and compute the vectors
document_vectors = {}

for group in groups:
    print(f"Processing group: {group}")
    
    # Get all the cleaned files for the group
    file_paths = get_file_paths(group)
    
    # Load all the documents for this group into a list (corpus)
    corpus = [load_text(file) for file in file_paths]
    
    # Compute TF-IDF matrix with BM25/Okapi heuristic
    bm25_matrix, vectorizer = compute_tfidf_bm25(corpus)
    
    # Reduce the matrix dimensions using SVD (optional, to avoid large sparse matrices)
    reduced_matrix = reduce_dimensions(bm25_matrix, n_components=100)
    
    # Store the vectors along with their corresponding document ids
    group_vectors = {}
    for idx, file_path in enumerate(file_paths):
        doc_id = os.path.basename(file_path).replace('_clean.txt', '')
        group_vectors[doc_id] = reduced_matrix[idx].tolist()  # Store as list for easier json serialization
    
    # Store the vectors for the current group
    document_vectors[group] = group_vectors

# Save the vectors to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(document_vectors, f, ensure_ascii=False, indent=4)

print(f"Document vectors saved to {output_file}")

