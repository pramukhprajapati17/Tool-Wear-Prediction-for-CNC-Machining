import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("dataset/tool_wear_prediction_dataset_5000.csv")

# Encode material type
material_map = {"Aluminum": 0, "Titanium": 1, "Steel": 2}
df["Material Type"] = df["Material Type"].map(material_map)

# Features and target
X = df.drop("Tool Wear (µm)", axis=1)
y = df["Tool Wear (µm)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
print("Model saved to models/rf_model.pkl")
