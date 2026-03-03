import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model():
    print("=== MODEL EVALUATION ===")

    X_test = np.load("artifacts/X_test.npy")
    y_test = np.load("artifacts/y_test.npy")
    model = joblib.load("artifacts/model.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClasses:", label_encoder.classes_)
    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate_model()
