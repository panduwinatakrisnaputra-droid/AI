from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Generate dataset with 2000 samples and 6 features
X, y = make_classification(
    n_samples=2000, 
    n_features=6,
    n_informative=3, 
    n_redundant=0,
    random_state=0, 
    shuffle=False
)

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

# Predict sample with 6 input feature values
p = clf.predict([[0, 0, 0, 0, 0, 0]])
print("Prediction:", p)