import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

# Base directory
base_dir = 'C:/Users/danie/Desktop/איחזור מידע/Information_retrieval/exe1/tf_idf_matrices'
output_file = os.path.join(base_dir, 'tfidf_feature_importance.xlsx')

# Group names
groups = ['A-J', 'BBC', 'J-P', 'NY-T']

# Function to load TF-IDF matrices from text files
def load_tfidf_matrix(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        matrix = [list(map(float, line.strip().split('\t'))) for line in lines]
    return np.array(matrix)

# Function to compute Information Gain
def compute_information_gain(matrix, labels):
    return mutual_info_classif(matrix, labels, discrete_features=False)

# Function to compute Chi-squared statistic
def compute_chi2(matrix, labels):
    # Normalize the matrix for Chi-squared
    scaler = MinMaxScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    chi2_scores, _ = chi2(matrix_scaled, labels)
    return chi2_scores

# Function to create labels based on row importance
def create_labels(matrix):
    row_importance = matrix.sum(axis=1)  # Calculate the sum of each row
    threshold = np.median(row_importance)  # Use the median as the threshold
    return (row_importance >= threshold).astype(int)  # Create binary labels

# Initialize Excel writer
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

# Process each group
for group in groups:
    print(f"Processing group: {group}")
    
    # Load TF-IDF matrices
    clean_file = os.path.join(base_dir, f"{group}_cleaned_matrix.txt")
    stem_file = os.path.join(base_dir, f"{group}_stemmed_matrix.txt")
    
    clean_matrix = load_tfidf_matrix(clean_file)
    stem_matrix = load_tfidf_matrix(stem_file)
    
    # Create labels dynamically for the group
    clean_labels = create_labels(clean_matrix)
    stem_labels = create_labels(stem_matrix)
    
    # Compute feature importance for cleaned TF-IDF matrix
    clean_info_gain = compute_information_gain(clean_matrix, clean_labels)
    clean_chi2 = compute_chi2(clean_matrix, clean_labels)
    
    # Compute feature importance for stemmed TF-IDF matrix
    stem_info_gain = compute_information_gain(stem_matrix, stem_labels)
    stem_chi2 = compute_chi2(stem_matrix, stem_labels)
    
    # Combine results for Cleaned and Stemmed TF-IDF matrices
    combined_results = pd.DataFrame({
        'Feature': range(clean_matrix.shape[1]),
        'Cleaned_Information Gain': clean_info_gain,
        'Cleaned_Chi-squared': clean_chi2,
        'Stemmed_Information Gain': stem_info_gain,
        'Stemmed_Chi-squared': stem_chi2
    })
    
    # Save results for the group to Excel
    combined_results.to_excel(writer, sheet_name=f"{group}_Results", index=False)

# Save Excel file
writer.close()
print(f"TF-IDF feature importance analysis saved to {output_file}")
