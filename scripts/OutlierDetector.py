from sklearn.cluster import KMeans
import torch

class OutlierDetector:

    def __init__(self, num_classes, latent_features, labels):
        self.num_classes = num_classes
        self.latent_features = latent_features
        self.labels = labels
        self.centroids = None
        self.distances = None
        self.outliers = None
        self.treshold = None

    def calculate_centroids(self):
        # Utilizza KMeans per calcolare i centroidi
        print(self.latent_features.shape)
        kmeans = KMeans(n_clusters=self.num_classes)
        kmeans.fit(self.latent_features.cpu().numpy())  # Converting to NumPy for compatibility with KMeans
        self.centroids = torch.tensor(kmeans.cluster_centers_, device=self.latent_features.device)
        return self.centroids

    def calculate_distances(self):
        # Calcola le distanze dalle caratteristiche latenti ai centroidi
        self.distances = torch.zeros(self.latent_features.shape[0], self.num_classes, device=self.latent_features.device)
        for i in range(self.num_classes):
            self.distances[:, i] = torch.norm(self.latent_features - self.centroids[i], dim=1)
        return self.distances
    
    def relabel_outliers(self):
        distances = self.calculate_distances()
        _, labels = torch.min(distances, dim=1)
        return labels
    
    def compute_treshold(self, threshold=None):
        if threshold is None:
            threshold = self.distances.mean() + 2 * self.distances.std()
        self.treshold = threshold
        return threshold
    
    def detect_outliers(self, threshold=None):
        # Calcola il centroide (media lungo ogni colonna)
        centroid = self.distances.mean(dim=0)

        # Calcola la matrice di covarianza
        cov_matrix = torch.cov(self.distances.T)

        # Inverti la matrice di covarianza
        cov_matrix_inv = torch.linalg.inv(cov_matrix)

        # Calcola la distanza di Mahalanobis per ciascun campione
        diffs = self.distances - centroid
        mahalanobis_distances = torch.sqrt(
            torch.sum(diffs @ cov_matrix_inv * diffs, dim=1)
        )
        # Identifica outlier in base alla soglia
        self.outliers = mahalanobis_distances > threshold
        return self.outliers

        
    def fit(self, threshold=None):
        '''
        This method calculates the centroids using KMeans, computes the distances
        of the latent features from the centroids, and calculates the outliers
        based on the threshold.
        The output is a tensor of boolean values that are True if the sample is an outlier
        and False otherwise.
        '''
        self.calculate_centroids()
        self.calculate_distances()
        threshold = self.compute_treshold()
        self.detect_outliers(threshold)
        return self.outliers
