# train_and_save_pipeline.py
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

# Columns
categorical_cols = ["sex", "smoker", "region"]
numerical_cols = ["age", "bmi", "children"]

# Preprocessor
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first"), categorical_cols),
    ("scale", StandardScaler(), numerical_cols)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit pipeline
pipeline.fit(X, y)

# Save the pipeline
with open("data_pickle_2.pkl", "wb") as f:
    pickle.dump(pipeline, f)
