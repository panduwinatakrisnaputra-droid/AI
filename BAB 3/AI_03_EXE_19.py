import pandas as pd
from sklearn.datasets import load_breast_cancer
from pycaret import classification

# 1. Load Breast Cancer dataset as DataFrame
cancer = load_breast_cancer(as_frame=True)
df = cancer.data
df['Target'] = cancer.target

# 2. Setup PyCaret Classification Pipeline
classification.setup(data=df, target='Target', session_id=123)

# 3. Evaluate and Compare Models
best_model = classification.compare_models()