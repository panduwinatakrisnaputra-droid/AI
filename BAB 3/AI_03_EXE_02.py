from sklearn import svm, datasets

iris = datasets.load_iris()

# Take 3rd and 4th features: Petal length (col index 2) and Petal width (col index 3)
X = iris.data[:, 2:4]
y = iris.target # 0: Setosa, 1: Versicolor, 2: Virginica

clf = svm.SVC()
clf.fit(X, y)

# Predict using petal length and width (e.g., 1.5 cm length, 0.3 cm width)
p = clf.predict([[1.5, 0.3]])
print("Prediction:", p)