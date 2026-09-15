from sklearn import svm

# 6 samples of [Height(cm), Weight(kg), ShoeSize(UK)]
X = [
    [170, 70, 10], 
    [180, 80, 12], 
    [170, 65, 8],
    [160, 55, 7],
    [175, 75, 9],
    [155, 50, 6]
]

# 6 target labels: 0 for Male, 1 for Female
y = [0, 0, 1, 1, 0, 1] 

clf = svm.SVC()
clf.fit(X, y)

# Predict for a test input
p = clf.predict([[160, 60, 7]])
print("Prediction:", p)