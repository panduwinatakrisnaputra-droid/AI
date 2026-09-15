import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 1. Generate non-linear sample data (Sine wave + noise)
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 2. Fit SVR model with Radial Basis Function (RBF) kernel
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X, y)

# 3. Predict across continuous range for smooth curve
X_plot = np.linspace(0, 5, 100)[:, np.newaxis]
y_pred = svr_rbf.predict(X_plot)

# 4. Visualisation
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='darkorange', label='Observed Data')
plt.plot(X_plot, y_pred, color='navy', lw=2, label='RBF SVR Model')
plt.xlabel("Input X")
plt.ylabel("Target Y")
plt.title("Support Vector Regression (SVR)")
plt.legend()
plt.grid(True)
plt.show()