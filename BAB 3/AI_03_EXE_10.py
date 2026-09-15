from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Model names
names = [
    "SVM", "Naive Bayes", "LDA", "QDA", 
    "Decision Tree", "Random Forest", "Nearest Neighbors", "Neural Networks"
]

# Classifier instances
classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)
]

# Load Diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Note: The target 'y' in load_diabetes is continuous (a regression task).
# To use classifiers, we convert target 'y' into binary classes (above/below median).
y_binary = (y > y.mean()).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.5, random_state=0
)

# Train and evaluate models
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name + ": " + str(score))