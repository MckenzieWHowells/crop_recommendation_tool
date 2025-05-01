"""
Crop Recommender System
This script trains a machine learning model to recommend crops based on soil and climate data.
"""

import os
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("crop_data.csv")

X = df.drop("label", axis=1)  # Features
y = df["label"]               # Target (crop)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

timestamp = int(time.time())

# Define relative path to models folder
model_path = f"models/crop_recommender_{timestamp}.pkl"

# Ensure the directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the model
joblib.dump(model, model_path)
print(f"Model trained and saved to: {model_path}")

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2%}")

# Save metadata
metadata = {
    "timestamp": timestamp,
    "accuracy": round(accuracy * 100, 2),
    "model_path": model_path
}

json_path = model_path.replace(".pkl", ".json")
with open(json_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {json_path}")

# Step 6: Predict (Example)
def recommend_crop():
    print("\nEnter your soil and climate data:")
    n = float(input("Nitrogen (N): "))
    p = float(input("Phosphorus (P): "))
    k = float(input("Potassium (K): "))
    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH level: "))
    rainfall = float(input("Rainfall (mm): "))

    input_data = [[n, p, k, temp, humidity, ph, rainfall]]
    crop = model.predict(input_data)[0]
    print(f"\nRecommended crop: **{crop.title()}**")

if __name__ == "__main__":
    # Load the model
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    
    # Example usage
recommend_crop()
