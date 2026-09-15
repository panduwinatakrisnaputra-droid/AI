from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the Linnerud multi-output dataset
data = load_linnerud()
X, y = data.data, data.target

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Fit multi-output Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Linnerud Feature Names:", data.feature_names)
print("Linnerud Target Names:", data.target_names)
print("Mean Squared Error (MSE):", round(mse, 4))