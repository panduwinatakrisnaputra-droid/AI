import pandas as pd
from flaml import AutoML
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. Fetch California Housing data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Train-test split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Instantiate AutoML Predictor
automl = AutoML()

# Define settings for the training
settings = {
    "time_budget": 60,      # Maximum time (in seconds) to let it search for models
    "metric": 'r2',         # The metric to optimize (R-squared for regression)
    "task": 'regression',   # Type of machine learning problem
    "verbose": 0            # Set to 1 or 2 if you want to see the progress in the console
}

# 4. Train the model
print("Training models...")
automl.fit(dataframe=df_train, label='MedHouseVal', **settings)

# 5. Evaluate the model
print(f"Best ML model found: {automl.best_estimator}")
print(f"R2 Score on test data: {automl.score(df_test, df_test['MedHouseVal'])}")