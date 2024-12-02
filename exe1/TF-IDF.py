import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Create directories to store the result
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1'
clean_dir = os.path.join(base_dir, 'clean_texts')
stem_dir = os.path.join(base_dir, 'stemmed_texts')

# Group names
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# Prepare the output directory to store the matrices
output_dir = os.path.join(base_dir, 'tf_idf_matrices')
os.makedirs(output_dir, exist_ok=True)

# Function to get all file paths in a directory for each group
def get_file_paths(group, text_type='clean'):
    if text_type == 'clean':
        group_dir = os.path.join(clean_dir, group)
    elif text_type == 'stemmed':
        group_dir = os.path.join(stem_dir, group)
    else:
        raise ValueError("text_type must be either 'clean' or 'stemmed'")
    
    return [os.path.join(group_dir, file) for file in os.listdir(group_dir) if file.endswith(f'_{text_type}.txt')]

# Load the text from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Apply TF-IDF and BM25/Okapi (heuristic adjustment)
def compute_tfidf_bm25(corpus):
    vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True, norm='l2', stop_words='english') 
    
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
    
    # Get all the cleaned and stemmed files for the group
    clean_file_paths = get_file_paths(group, text_type='clean')
    stem_file_paths = get_file_paths(group, text_type='stemmed')
    
    # Load all the documents for each group into a list (corpus)
    clean_corpus = [load_text(file) for file in clean_file_paths]
    stem_corpus = [load_text(file) for file in stem_file_paths]
    
    # Compute TF-IDF matrix with BM25/Okapi heuristic for cleaned corpus
    bm25_clean_matrix, vectorizer_clean = compute_tfidf_bm25(clean_corpus)
    reduced_clean_matrix = reduce_dimensions(bm25_clean_matrix, n_components=100)
    
    # Compute TF-IDF matrix with BM25/Okapi heuristic for stemmed corpus
    bm25_stem_matrix, vectorizer_stem = compute_tfidf_bm25(stem_corpus)
    reduced_stem_matrix = reduce_dimensions(bm25_stem_matrix, n_components=100)
    
    # Convert the matrices into a readable format (list of lists)
    clean_matrix_list = reduced_clean_matrix.tolist()
    stem_matrix_list = reduced_stem_matrix.tolist()
    
    # Prepare output filenames
    clean_output_file = os.path.join(output_dir, f"{group}_cleaned_matrix.txt")
    stem_output_file = os.path.join(output_dir, f"{group}_stemmed_matrix.txt")
    
    # Save the clean matrix to a text file (tab-delimited format for easier readability)
    with open(clean_output_file, 'w', encoding='utf-8') as f:
        for row in clean_matrix_list:
            f.write("\t".join(map(str, row)) + "\n")  # Tab-delimited format
    
    # Save the stemmed matrix to a text file (tab-delimited format for easier readability)
    with open(stem_output_file, 'w', encoding='utf-8') as f:
        for row in stem_matrix_list:
            f.write("\t".join(map(str, row)) + "\n")  # Tab-delimited format

    # Store the vectors for the current group
    document_vectors[group] = {
        'clean': clean_matrix_list,
        'stem': stem_matrix_list
    }

print("Matrices have been saved as text files.")
