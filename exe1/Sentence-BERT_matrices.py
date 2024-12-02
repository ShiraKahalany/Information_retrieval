import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Base directory and paths
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1'
output_dir = os.path.join(base_dir, 'Sentence-BERT_matrices')  # תיקיה לשמירת הווקטורים
os.makedirs(output_dir, exist_ok=True)

# קובץ ה-Excel
excel_file = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1/posts_first_targil.xlsx'

# Group names (sheets in the Excel file)
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# טוען את המודל Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')  # דגם קל ומהיר

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
    
    # Create SBERT vectors for each document
    document_vectors = []
    for text in sheet_data['Body Text']:
        vector = model.encode(str(text), convert_to_numpy=True)
        document_vectors.append(vector)
    
    # Save the document vectors to a text file
    output_file = os.path.join(output_dir, f"{group}_Sentence-BERT_vectors.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for vector in document_vectors:
            f.write("\t".join(map(str, vector)) + "\n")
    
    print(f"Saved vectors for group: {group}")

print("Sentence-BERT document vectors have been saved.")
