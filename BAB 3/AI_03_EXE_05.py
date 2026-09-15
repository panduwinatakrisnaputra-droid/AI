from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import joblib

# Load data
X, y = load_iris(return_X_y=True)

# Train model
clf = GaussianNB()
clf.fit(X, y)

# Serialize (Save) the model
joblib.dump(clf, 'iris_nb_model.pkl')
print("Model successfully saved to 'iris_nb_model.pkl'.")

# Deserialize (Load) the model back
clf_loaded = joblib.load('iris_nb_model.pkl')
print("Model successfully loaded.")

# Make prediction with the loaded model
p = clf_loaded.predict([[5.0, 3.4, 1.5, 0.4]])
print("Prediction from loaded model:", p)