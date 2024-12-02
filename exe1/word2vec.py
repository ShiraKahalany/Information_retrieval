import os
import re
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings 
import nltk

# Download the NLTK resources
nltk.download('stopwords')

warnings.filterwarnings(action='ignore')

# Base directory and paths
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1'
clean_dir = os.path.join(base_dir, 'clean_texts')
stem_dir = os.path.join(base_dir, 'stemmed_texts')

# Output directory for word2vec vectors
output_dir = os.path.join(base_dir, 'word2vec_matrices')
os.makedirs(output_dir, exist_ok=True)

# Group names
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# Function to get all file paths in a directory for each group
def get_file_paths(group, text_type='clean'):
    group_dir = os.path.join(clean_dir, group) if text_type == 'clean' else os.path.join(stem_dir, group)
    return [os.path.join(group_dir, file) for file in os.listdir(group_dir) if file.endswith(f'_{text_type}.txt')]

# Load the text from a file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Function to clean text (remove punctuation, numbers, stopwords)
def clean_text(text):
    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Function to create document vector by averaging word vectors
def create_document_vector(tokens, model):
    word_vectors = []
    
    for token in tokens:
        if token in model.wv:  # Check if the word exists in the model's vocabulary
            word_vectors.append(model.wv[token])
    
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)  # Return zero vector if no valid words found
    
    return np.mean(word_vectors, axis=0)  # Average the word vectors to represent the document

# Process each group and create document vectors
for group in groups:
    print(f"Processing group: {group}")
    
    # Get file paths for clean and stemmed texts
    clean_file_paths = get_file_paths(group, text_type='clean')
    stem_file_paths = get_file_paths(group, text_type='stemmed')
    
    # Load and clean texts for each group
    clean_corpus = [clean_text(load_text(file)) for file in clean_file_paths]
    stem_corpus = [clean_text(load_text(file)) for file in stem_file_paths]
    
    # Train Word2Vec model on the combined clean and stemmed corpora
    combined_corpus = clean_corpus + stem_corpus
    #HERE HE MAKES THE VECTORS LOCALLY
    model = Word2Vec(sentences=combined_corpus, vector_size=100, window=5, min_count=1, sg=1)
    
    # Create document vectors for clean corpus
    clean_vectors = []
    for tokens in clean_corpus:
        clean_vectors.append(create_document_vector(tokens, model))
    
    # Create document vectors for stemmed corpus
    stem_vectors = []
    for tokens in stem_corpus:
        stem_vectors.append(create_document_vector(tokens, model))
    
    # Save the clean and stemmed document vectors to text files
    clean_output_file = os.path.join(output_dir, f"{group}_cleaned_word2vec_vectors.txt")
    stem_output_file = os.path.join(output_dir, f"{group}_stemmed_word2vec_vectors.txt")
    
    # Save clean vectors
    with open(clean_output_file, 'w', encoding='utf-8') as f:
        for vector in clean_vectors:
            f.write("\t".join(map(str, vector)) + "\n")
    
    # Save stemmed vectors
    with open(stem_output_file, 'w', encoding='utf-8') as f:
        for vector in stem_vectors:
            f.write("\t".join(map(str, vector)) + "\n")

print("Word2Vec document vectors have been saved as text files.")
