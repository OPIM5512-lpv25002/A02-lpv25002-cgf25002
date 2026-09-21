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

# Predict on training data
train_pred = mlp_regressor.predict(X_train)

# Make sure figures folder exists
os.makedirs("figures", exist_ok=True)

# Actual vs. Predicted - Training Data
plt.figure(figsize=(7, 6))
plt.scatter(y_train, train_pred, alpha=0.4)

# Reference line for perfect predictions
min_val = min(y_train.min(), train_pred.min())
max_val = max(y_train.max(), train_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs. Predicted - Training Data")

plt.tight_layout()
plt.savefig("figures/train_actual_vs_pred.png", dpi=150)
plt.close()

print("Saved figures/train_actual_vs_pred.png")