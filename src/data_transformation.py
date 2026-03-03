import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def transform_data():
    train = pd.read_csv("artifacts/train.csv")
    test = pd.read_csv("artifacts/test.csv")

    print("=== DATA TRANSFORMATION ===")

    # Separate features and label
    X_train = train.drop("label", axis=1)
    y_train = train["label"]
    X_test = test.drop("label", axis=1)
    y_test = test["label"]

    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        # Handle unseen labels in test
        X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Save everything
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(encoders, "artifacts/encoders.pkl")
    joblib.dump(label_encoder, "artifacts/label_encoder.pkl")

    np.save("artifacts/X_train.npy", X_train_scaled)
    np.save("artifacts/X_test.npy", X_test_scaled)
    np.save("artifacts/y_train.npy", y_train_enc)
    np.save("artifacts/y_test.npy", y_test_enc)

    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape: {X_test_scaled.shape}")
    print(f"Classes: {label_encoder.classes_}")
    print("Transformation complete.")

if __name__ == "__main__":
    transform_data()
