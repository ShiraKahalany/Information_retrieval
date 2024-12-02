import os
import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk resources are available
nltk.download('punkt')

# Create directories to store the results
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1'
clean_dir = os.path.join(base_dir, 'clean_texts')
stem_dir = os.path.join(base_dir, 'stemmed_texts')

# List of groups to process
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# Create subdirectories for each group
for group in groups:
    os.makedirs(os.path.join(clean_dir, group), exist_ok=True)
    os.makedirs(os.path.join(stem_dir, group), exist_ok=True)

# Load the Excel file
file_path = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1/posts_first_targil.xlsx'
data = pd.ExcelFile(file_path)

# Initialize stemmer
stemmer = PorterStemmer()

def clean_text(text):
    """Removes punctuation by separating words and punctuation."""
    return re.sub(r'(\w)([^\w\s])', r'\1 \2', re.sub(r'([^\w\s])(\w)', r'\1 \2', text))

def stem_text(text):
    """Applies stemming to the text."""
    tokens = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_words)

def process_sheet(sheet_name, data):
    """Processes a sheet: cleans and stems text, and saves results."""
    sheet_data = data.parse(sheet_name)
    
    if 'Body Text' not in sheet_data.columns:
        print(f"Skipping {sheet_name}: 'Body Text' column missing.")
        return

    # Drop rows with missing 'Body Text'
    sheet_data = sheet_data.dropna(subset=['Body Text'])

    for idx, row in sheet_data.iterrows():
        doc_id = f"{sheet_name}_{idx}"
        body_text = row['Body Text']

        # Clean text
        cleaned = clean_text(body_text)
        with open(os.path.join(clean_dir, sheet_name, f"{doc_id}_clean.txt"), 'w', encoding='utf-8') as f:
            f.write(cleaned)

        # Stem text
        stemmed = stem_text(cleaned)
        with open(os.path.join(stem_dir, sheet_name, f"{doc_id}_stemmed.txt"), 'w', encoding='utf-8') as f:
            f.write(stemmed)

# Process all specified groups
for group in groups:
    process_sheet(group, data)

print(f"Text processing completed. Cleaned and stemmed text files are saved in {base_dir}.")
