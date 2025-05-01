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

MODEL_DIR = "models"
DATA_PATH = "data/crop_recommendation_dataset_kaggle.csv"


def load_data(filepath: str):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)


def train_model(X, y):
    """Train a Random Forest model on the provided data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy


def save_model(model, directory: str, timestamp: int):
    """Save the trained model to a .pkl file with timestamp."""
    os.makedirs(directory, exist_ok=True)
    model_filename = f"crop_recommender_{timestamp}.pkl"
    model_path = os.path.join(directory, model_filename)
    joblib.dump(model, model_path)
    return model_path


def save_metadata(model_path: str, accuracy: float, timestamp: int):
    """Save metadata as a JSON file alongside the model."""
    metadata = {
        "timestamp": timestamp,
        "accuracy": round(accuracy * 100, 2),
        "model_path": model_path
    }
    json_path = model_path.replace(".pkl", ".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return json_path


def recommend_crop(model):
    """Prompt user for input and recommend a crop based on the model."""
    print("\nEnter your soil and climate data:")
    try:
        n = float(input("Nitrogen (N): "))
        p = float(input("Phosphorus (P): "))
        k = float(input("Potassium (K): "))
        temp = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH level: "))
        rainfall = float(input("Rainfall (mm): "))
    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
        return

    input_data = [[n, p, k, temp, humidity, ph, rainfall]]
    crop = model.predict(input_data)[0]
    print(f"\nRecommended crop: **{crop.title()}**")


def main():
    df = load_data(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]

    model, accuracy = train_model(X, y)
    timestamp = int(time.time())

    model_path = save_model(model, MODEL_DIR, timestamp)
    print(f"✅ Model trained and saved to: {model_path}")

    json_path = save_metadata(model_path, accuracy, timestamp)
    print(f"📄 Metadata saved to: {json_path}")
    print(f"📊 Model accuracy: {accuracy:.2%}")

    # Load model for prediction
    loaded_model = joblib.load(model_path)
    print("📦 Model loaded for prediction.")
    recommend_crop(loaded_model)


if __name__ == "__main__":
    main()
