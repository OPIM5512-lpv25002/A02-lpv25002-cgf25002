from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import os

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
from sklearn.model_selection import train_test_split

X = df.drop('MedHouseVal', axis=1)  # Features
y = df['MedHouseVal']              # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neural_network import MLPRegressor

# Initialize MLPRegressor with early stopping and custom hyperparameters
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=42)

# Train the model
mlp_regressor.fit(X_train, y_train)
print("MLPRegressor trained successfully!")