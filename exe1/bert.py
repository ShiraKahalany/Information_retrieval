import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Base directory and paths
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1'
output_dir = os.path.join(base_dir, 'bert_matrices')  # תיקיה לשמירת הווקטורים
os.makedirs(output_dir, exist_ok=True)

# קובץ ה-Excel
excel_file = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1/posts_first_targil.xlsx'

# Group names (sheets in the Excel file)
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# הגדרת המודל והטוקניזר של BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to clean and truncate text (max 512 tokens for BERT)
def prepare_text(text):
    # Tokenize and truncate to 512 tokens
    encoded = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    return encoded

# Function to create BERT embeddings for a document
def create_bert_vector(encoded_input):
    with torch.no_grad():
        outputs = model(**encoded_input)
        # Using the [CLS] token embedding as the document representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token
    return cls_embedding.squeeze().numpy()

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
    
    # Create BERT vectors for each document
    document_vectors = []
    for text in sheet_data['Body Text']:
        encoded_input = prepare_text(str(text))
        vector = create_bert_vector(encoded_input)
        document_vectors.append(vector)
    
    # Save the document vectors to a text file
    output_file = os.path.join(output_dir, f"{group}_bert_vectors.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for vector in document_vectors:
            f.write("\t".join(map(str, vector)) + "\n")
    
    print(f"Saved vectors for group: {group}")

print("BERT document vectors have been saved.")
