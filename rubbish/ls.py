import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def plot_results(reduced_data, labels, title, save_path=None):
    """
    Plot the reduced dimensional data, colored by labels, and save the plot to a file if a path is provided.

    Args:
        reduced_data (np.ndarray): Reduced data points, expected to be 2D.
        labels (np.ndarray): Labels for coloring the points.
        title (str): Title of the plot.
        save_path (str, optional): The path where the plot will be saved. If None, the plot will not be saved. Defaults to None.
    """
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('viridis', len(unique_labels))

    plt.figure(figsize=(20, 10))
    plt.xlim(-30, 30)
    plt.ylim(-5, 5)
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=colors(i), label=f'Label {label}')
    plt.title(title)
    plt.xlabel('Principal Component 1' if title.startswith('PCA') else 'Dimension 1')
    plt.ylabel('Principal Component 2' if title.startswith('PCA') else 'Dimension 2')
    plt.legend()

    if save_path:
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()



with open("/CondiPDiff/continue-latent-tsne-dense.data", "rb") as f:
    diction = pickle.load(file=f)
plot_results(**diction, save_path="continue-latent-tsne-dense.jpg")