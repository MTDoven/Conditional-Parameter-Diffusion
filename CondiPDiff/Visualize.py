from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def visualize_data(tensor_data, labels, pca=True, tsne=True, save_path=None):
    # Flatten the tensor except for the batch dimension
    flattened_data = tensor_data.view(tensor_data.size(0), -1).cpu().numpy()
    labels = labels.numpy()  # Ensure labels are also converted to numpy array if they aren't already

    if pca:
        pca_model = PCA(n_components=2)
        pca_results = pca_model.fit_transform(flattened_data)
        plot_results(pca_results, labels, "PCA Visualization", save_path)

    if tsne:
        tsne_model = TSNE(n_components=2, random_state=42, verbose=0, early_exaggeration=12, perplexity=40)
        tsne_results = tsne_model.fit_transform(flattened_data)
        plot_results(tsne_results, labels, "t-SNE Visualization", save_path)


def plot_results(reduced_data, labels, title, save_path=None, x_min_max_y_min_max=(-30, 30, -5, 5)):
    diction = {"reduced_data": reduced_data, "labels": labels, "title": title}
    with open(save_path+".data", "wb") as f:
        pickle.dump(diction, file=f)

    unique_labels = np.unique(labels)
    colors = plt.get_cmap('viridis', len(unique_labels))

    plt.figure(figsize=(10, 10))
    plt.xlim(x_min_max_y_min_max[0], x_min_max_y_min_max[1])
    plt.ylim(x_min_max_y_min_max[2], x_min_max_y_min_max[3])
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=colors(i), label=f'Label {label}')
    plt.title(title)
    plt.xlabel('Principal Component 1' if title.startswith('PCA') else 'Dimension 1')
    plt.ylabel('Principal Component 2' if title.startswith('PCA') else 'Dimension 2')
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    with open("/home/wangkai/cpdiff/condipdiff/CondiPDiff/continue-latent-tsne-sparse.data", "rb") as f:
        diction = pickle.load(file=f)
    plot_results(**diction, save_path="./result.jpg")