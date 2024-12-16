import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Base directory
base_dir = 'C:/Users/danie/Desktop/IR/Information_retrieval/exe2/IR-files'
output_dir = 'C:/Users/danie/Desktop/IR/Information_retrieval/exe2/clustering_results'
os.makedirs(output_dir, exist_ok=True)

def load_matrix(file_path):
    """Load a numeric matrix from a file (CSV or Excel)."""
    try:
        print(f"Loading file: {file_path}")
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, header=None).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, header=None).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
        else:
            print(f"Unsupported file format: {file_path}")
            return np.array([])
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return np.array([])

def combine_matrices(folder_name, base_dir):
    """Load and combine matrices dynamically to avoid memory overflow."""
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return np.array([])
    
    combined_matrix = None
    min_columns = float('inf')
    print(f"Processing folder: {folder_path}")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(('.csv', '.xlsx')):
            matrix = load_matrix(file_path)
            if matrix.size > 0:
                min_columns = min(min_columns, matrix.shape[1])
                if combined_matrix is None:
                    combined_matrix = matrix[:, :min_columns]
                else:
                    combined_matrix = np.vstack((combined_matrix, matrix[:, :min_columns]))
    
    if combined_matrix is None or combined_matrix.size == 0:
        raise ValueError(f"No valid data in folder: {folder_name}")
    return combined_matrix

def reduce_dimensions(data, max_dimensions=50):
    """Reduce dimensions using PCA if necessary."""
    if data.shape[1] > max_dimensions:
        print(f"Reducing dimensions from {data.shape[1]} to {max_dimensions}")
        return PCA(n_components=max_dimensions, random_state=42).fit_transform(data)
    return data

def plot_clusters(data, labels, title, output_file):
    """Visualize clusters using t-SNE."""
    print("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, metric='cosine')
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    for cluster in np.unique(labels):
        mask = labels == cluster
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], label=f"Cluster {cluster}", alpha=0.7)

    plt.title(title)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def run_clustering_and_save(group, combined_matrix):
    """Run clustering algorithms and save the results."""
    print(f"Running clustering for group: {group}")
    # Reduce dimensions
    data = reduce_dimensions(combined_matrix)

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    plot_clusters(data, kmeans_labels, f"K-Means Clustering - {group}", os.path.join(output_dir, f"{group}_kmeans_clusters.png"))

    # DBSCAN with dynamic eps
    eps = max(0.5, np.median(pdist(data)) * 0.1)
    dbscan = DBSCAN(eps=eps, min_samples=5, metric='cosine')
    dbscan_labels = dbscan.fit_predict(data)
    plot_clusters(data, dbscan_labels, f"DBSCAN Clustering - {group}", os.path.join(output_dir, f"{group}_dbscan_clusters.png"))

    # GMM
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    plot_clusters(data, gmm_labels, f"Gaussian Mixture Clustering - {group}", os.path.join(output_dir, f"{group}_gmm_clusters.png"))

    # Save results
    results_df = pd.DataFrame({
        'Document': range(data.shape[0]),
        'KMeans_Cluster': kmeans_labels,
        'DBSCAN_Cluster': dbscan_labels,
        'GMM_Cluster': gmm_labels
    })
    results_file = os.path.join(output_dir, f"{group}_clustering_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

# Main process
groups = ['bert-sbert', 'doc2vec', 'glove', 'word2vec']
for group in groups:
    try:
        combined_matrix = combine_matrices(group, base_dir)
        print(f"Combined matrix shape for {group}: {combined_matrix.shape}")
        run_clustering_and_save(group, combined_matrix)
    except ValueError as e:
        print(e)
        continue

print("Clustering tasks completed successfully.")
