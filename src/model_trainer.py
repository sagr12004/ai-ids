import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def train_model():
    print("=== MODEL TRAINING ===")

    X_train = np.load("artifacts/X_train.npy")
    y_train = np.load("artifacts/y_train.npy")

    print(f"Training on {X_train.shape[0]} samples...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")

    print("Model trained and saved to artifacts/model.pkl")

if __name__ == "__main__":
    train_model()
