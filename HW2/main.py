from typing import Tuple, List

import click
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import cv2
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection
)
import matplotlib.pyplot as plt
import plotly.express as px



LABELS_MAPPING = {
    "human": 0,
    "animal": 1
}

RANDOM_SEED = 42

def load_data(image_folder: str, label_file: str) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    ''' Loads images and labels from the specified folder and file.'''
    # load labels file
    metadata = pd.read_csv(label_file, sep="|")
    labels = metadata["label"].map(LABELS_MAPPING).values
    descriptions = metadata["comment"].values

    # load corresponding images
    images = []
    
    for image_path in metadata["image_name"].values:
        image = cv2.imread(f"{image_folder}/{image_path}") # HxWxC in BGR format
        images.append(image)

    return images, labels, descriptions

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None

        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # standardize data
        self.standardize(X)

        # calculate covariance matrix
        cov_matrix = np.cov(X.T)

        # get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # sort components
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # reduce data using number of components (n_components)
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # standardize data
        X = self.standardize(X)

        # reduce data using number of components
        X_reduced = np.dot(X, self.components)

        return X_reduced

    def standardize(self, X: np.ndarray) -> np.ndarray:
        ''' Normalize data by substracting mean and divide by std '''
        X_norm = (X - self.mean) / self.std
        return X_norm


class KMeans:
    def __init__(self, n_clusters: int, max_iterations: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iterations
        self.centroids = None

    def fit(self, X: np.ndarray, seed: int = None) -> None:
        np.random.seed(seed=seed)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            clusters = self.assign_clusters(self.centroids, X)
            new_centroids = self.compute_means(clusters, X)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.assign_clusters(self.centroids, X)

    def assign_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        distances = np.array([self.euclidean_distance(x, centroids) for x in X])
        return np.argmin(distances, axis=1)

    def compute_means(self, clusters: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.array([X[clusters == k].mean(axis=0) for k in range(self.n_clusters)])

    def euclidean_distance(self, a, b) -> float:
        return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def vectorize_images(images_list: List[np.ndarray]) -> np.ndarray:
    return np.load("vectorized_images.npy")
    # model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # embeddings = []
    # for i in range(len(images_list)):
    #     image = images_list[i]
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #     image = np.expand_dims(image, axis=0)  # Add batch dimension
        
    #     inputs = processor(images=image, return_tensors="pt")
    #     outputs = model(**inputs)
    #     image_embeds = outputs.image_embeds.detach().cpu().numpy()
    #     embeddings.append(image_embeds)
    # embeddings = np.vstack(embeddings)
    # np.save("vectorized_images.npy", embeddings)
    # return embeddings

def vectorize_text(descriptions: List[str]) -> np.ndarray:
    return np.load("vectorized_descriptions.npy")

    # model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # inputs = tokenizer(descriptions, padding=True, return_tensors="pt")
    # outputs = model(**inputs)
    # embeddings = outputs.text_embeds.detach().cpu().numpy()

    # np.save("vectorized_descriptions.npy", embeddings)
    # return embeddings


def visualize_drv_images(
    drv_images: np.ndarray,
    labels: np.ndarray,
    plot_name: str = None,
    algorithm_name: str = "PCA",
    title: str = None,
) -> None:
    if np.iscomplexobj(drv_images):
        drv_images = np.real(drv_images.copy())
    
    if drv_images.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(drv_images[:, 0], drv_images[:, 1], c=labels, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label='Label')
        plt.xlabel(f"{algorithm_name} Component 1")
        plt.ylabel(f"{algorithm_name} Component 2")
        if title:
            plt.title(title)

        plt.savefig(f"plots/{plot_name}_plot.png", dpi=300)
        plt.close()
    elif drv_images.shape[1] == 3:
        fig = px.scatter_3d(
            x=drv_images[:, 0],
            y=drv_images[:, 1],
            z=drv_images[:, 2],
            color=labels,
            labels={'color': 'Label'},
            title=title if title else f"{algorithm_name} 3D Plot",
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{algorithm_name} Component 1",
                yaxis_title=f"{algorithm_name} Component 2",
                zaxis_title=f"{algorithm_name} Component 3"
            )
        )

        fig.write_html(f"plots/{plot_name}_plot.html")
    else:
        print("The number of components > 3 is not supported yet")


@click.command()
@click.option('--images_folder', type=str, help='Path to the input data')
@click.option('--labels_path', type=str, help='Path to the annotations')
@click.option('--n_components', type=int, help='Number of components')
@click.option('--n_clusters', type=int, help='Number of clusters')
def main(images_folder, labels_path, n_components, n_clusters):
    # load image data and text labels
    images, labels, descriptions = load_data(images_folder, labels_path)

    # vectorize images and text labels
    vImages = vectorize_images(images)
    
    # PCA or t-SNE on images
    
    dimred = PCA(n_components=n_components)
    dimred.fit(vImages)
    drvImages = dimred.transform(vImages)
    
    # dimred = TSNE(n_components=n_components)
    # drvImages = dimred.fit_transform(vImages)

    # Visualize 2D and 3D embeddings of images and color points based on labels
    visualize_drv_images(
        drv_images=drvImages,
        labels=labels,
        plot_name=f"PCA_labels_{n_components}_components",
        algorithm_name="PCA",
        title="Human is 0, animal is 1"
    )

    # Perform clustering on the embeddings and visualize the results
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(np.real(drvImages))

    # clusterer = KMeans(n_clusters=n_clusters)
    # clusterer.fit(vImages, seed=RANDOM_SEED)
    # cluster_labels = clusterer.predict(vImages)

    # Visualize 2D and 3D embeddings of images and color points based on cluster label and original labels
    visualize_drv_images(
        drv_images=drvImages,
        labels=cluster_labels,
        plot_name=f"agg_clustering_on_3_PCA_components_{n_clusters}_clusters",
        algorithm_name="PCA",
        title="Agglomerative clustering algorithm trained on 3 PCA components"
    )

    # DBSCAN outlier detection
    EPSILON = 0.8
    clusterer = DBSCAN(eps=EPSILON)
    cluster_labels = clusterer.fit_predict(np.real(drvImages))

    visualize_drv_images(
        drv_images=drvImages,
        labels=cluster_labels,
        plot_name=f"DBSCAN_with_epsilon_{EPSILON}",
        algorithm_name="PCA",
        title="DBSCAN trained on 3 PCA components"
    )


    # Select few text descriptions and select nearest neighbors based on embeddings. 
    vText = vectorize_text(descriptions.tolist())
    drvText = dimred.transform(vText)
    drvText = np.real(drvText)

    np.random.seed(RANDOM_SEED)
    chosen_samples = np.random.choice(drvText.shape[0], 5, replace=False)

    N_NEIGHBORS = 5

    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1)  # +1 because the sample itself will be included
    nbrs.fit(drvText)

    for ind in chosen_samples:
        distances, indices = nbrs.kneighbors(drvText[ind].reshape(1, -1))
        
        fig, axes = plt.subplots(1, N_NEIGHBORS + 1, figsize=(15, 5))

        original_image = images[ind]
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        for i in range(1, N_NEIGHBORS + 1):
            neighbor_image = images[indices[0][i]]
            neighbor_image = cv2.cvtColor(neighbor_image, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(neighbor_image)
            axes[i].set_title(f"Neighbor {i}")
            axes[i].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"plots/neighbours/sample_{ind}.png", dpi=300)
        plt.close(fig)
    

if __name__ == "__main__":
    main()