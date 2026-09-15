from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load 4-dimensional Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Reduce features from 4D down to 2D using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 3. Plot 2D projection
plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Set1', edgecolors='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Dimensionality Reduction (4D -> 2D)")
plt.colorbar(scatter, ticks=[0, 1, 2], label='Iris Species')
plt.grid(True)
plt.show()