import pandas as pd

def validate_data():
    train = pd.read_csv("artifacts/train.csv")
    test = pd.read_csv("artifacts/test.csv")

    print("=== DATA VALIDATION ===")

    # Check missing values
    train_missing = train.isnull().sum().sum()
    test_missing = test.isnull().sum().sum()
    print(f"Train missing values: {train_missing}")
    print(f"Test missing values: {test_missing}")

    # Check column count
    print(f"Train columns: {train.shape[1]}")
    print(f"Test columns: {test.shape[1]}")

    # Check duplicates
    train_dups = train.duplicated().sum()
    test_dups = test.duplicated().sum()
    print(f"Train duplicates: {train_dups}")
    print(f"Test duplicates: {test_dups}")

    # Check label distribution
    print("\nTrain label distribution:")
    print(train["label"].value_counts())

    print("\nValidation complete.")
    return train, test

if __name__ == "__main__":
    validate_data()
