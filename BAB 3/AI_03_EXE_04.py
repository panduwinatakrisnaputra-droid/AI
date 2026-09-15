import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Select four specified feature columns
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
df[selected_features].hist(bins=20, figsize=(10, 8))

plt.suptitle('Histograms of Breast Cancer Features')
plt.tight_layout()
plt.show()