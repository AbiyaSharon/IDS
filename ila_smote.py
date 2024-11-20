import numpy as np
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE 

# Adaptive SMOTE function
def adaptive_smote(X_minority, n_samples, k_neighbors=5):
    n_minority_samples = X_minority.shape[0]  # Number of minority samples
    n_features = X_minority.shape[1]  # Correctly set the number of features
    # Nearest Neighbors model for minority class
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X_minority)
    indices = nbrs.kneighbors(X_minority, return_distance=False)[:, 1:]  # Exclude self-neighbor
    synthetic_samples = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # Randomly choose a minority class sample
        idx = np.random.randint(0, n_minority_samples)
        x_i = X_minority[idx]
        # Randomly select one of the k-nearest neighbors
        neighbor_idx = np.random.choice(indices[idx])
        x_neighbor = X_minority[neighbor_idx]
        # Generate synthetic point in the direction of the neighbor
        alpha = np.random.rand()  # Random weight factor
        synthetic_samples[i] = x_i + alpha * (x_neighbor - x_i)
    return synthetic_samples

# ILA_SMOTE function
def ILA_SMOTE(X, y, m_cl=1, n_samples=100, k_neighbors=5):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled





