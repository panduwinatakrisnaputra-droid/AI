import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0: Malignant, 1: Benign

# Apply PCA to reduce 30 features down to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Breast Cancer Dataset')
plt.grid(True)
plt.show()