from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load wine dataset
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

N = y_test.shape[0]
C = (y_test == y_pred).sum()
print(f"Total points: {N} | Correctly labeled points: {C} | Accuracy: {C/N:.2%}")