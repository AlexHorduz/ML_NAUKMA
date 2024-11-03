from typing import Tuple, List

import click
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import pandas as pd
import cv2
from transformers import AutoProcessor, CLIPVisionModelWithProjection
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
    
    # for image_path in metadata["image_name"].values:
    #     image = cv2.imread(f"{image_folder}/{image_path}") # HxWxC in BGR format
    #     images.append(image)

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
    def __init__(self, n_clusters: int, max_iterations: int):
        self.n_clusters = n_clusters
        self.max_iter = max_iterations

        # randomly initialize cluster centroids
        self.centroids = None

    def fit(self, X: np.ndarray) -> None:
        for _ in range(self.max_iter):
            # create clusters by assigning the samples to the nearest centroids
            clusters = self.assign_clusters(self.centroids, X)
            # update centroids
            new_centroids = self.compute_means(clusters, X)


    def predict(self, X: np.ndarray) -> np.ndarray:
        # for each sample search for nearest centroids
        pass

    def assign_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' given input data X and cluster centroids assign clusters to samples '''
        pass

    def compute_means(self, clusters: np.ndarray, X: np.ndarray) -> np.ndarray:
        ''' recompute cluster centroids'''
        pass

    def euclidean_distance(self, a, b) -> float:
        """ Calculates the euclidean distance between two vectors a and b """
        return np.sqrt(np.sum(np.power(a - b, 2)))


def vectorize(images_list: List[np.ndarray]) -> np.ndarray:
    return np.load("vectorized_images.npy")
    # model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # images = []
    # for i in range(len(images_list)):
    #     # print(i)
    #     image = images_list[i]
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    #     image = np.expand_dims(image, axis=0)  # Add batch dimension
        
    #     inputs = processor(images=image, return_tensors="pt")
    #     outputs = model(**inputs)
    #     image_embeds = outputs.image_embeds.detach().cpu().numpy()
    #     images.append(image_embeds)
    # images = np.vstack(images)
    # np.save("vectorized_images.npy", images)
    # return images

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

def neighbour_search(
        text_req: np.ndarray, 
        vImages: np.ndarray, 
        top_k: int = 5
    ) -> np.ndarray:
    ''' search for top_k nearest neightbours in the space '''
    pass


@click.command()
@click.option('--images_folder', type=str, help='Path to the input data')
@click.option('--labels_path', type=str, help='Path to the annotations')
@click.option('--n_components', type=int, help='Number of components')
@click.option('--n_clusters', type=int, help='Number of clusters')
def main(images_folder, labels_path, n_components, n_clusters):
    # load image data and text labels
    images, labels, descriptions = load_data(images_folder, labels_path)

    # vectorize images and text labels
    vImages = vectorize(images)
    print(vImages.shape)
    
    # PCA or t-SNE on images
    # dimred = TSNE(n_components=n_components)
    dimred = PCA(n_components=n_components)
    dimred.fit(vImages)

    drvImages = dimred.transform(vImages)

    print(drvImages.shape)

    # Visualize 2D and 3D embeddings of images and color points based on labels
    visualize_drv_images(
        drv_images=drvImages,
        labels=labels,
        plot_name=f"PCA_labels_{n_components}_components",
        algorithm_name="PCA",
        title="Human is 0, animal is 1"
    )

    return 0
    # Perform clustering on the embeddings and visualize the results
    # clustere = AgglomerativeClustering(n_clusters=n_clusters)
    clusterer = KMeans(n_clusters=n_clusters)

    # Visualize 2D and 3D embeddings of images and color points based on cluster label and original labels
    # TODO

    # DBSCAN outlier detection
    clusterer = DBSCAN(eps=None)  # select good eps value !!!

    # Create a copy of your trained data with cleaned outliers 
    # TODO


    # Select few text descriptions and select nearest neighbors based on embeddings. 
    vText = vectorize(descriptions)
    drvText = dimred.transform(vText)
    # TODO 

    # Plot the results: text description, few nearest images
    # TODO


    

if __name__ == "__main__":
    main()