import pandas as pd
import requests
import random

# Load test data
df = pd.read_csv("artifacts/test.csv")

# Pick 10 random samples
samples = df.sample(10, random_state=42)

print("Testing 10 random samples from real dataset...\n")
print(f"{'#':<4} {'Actual':<10} {'Predicted':<12} {'Risk':<8} {'Match'}")
print("-" * 45)

for i, (_, row) in enumerate(samples.iterrows()):
    actual = row["label"]
    input_data = row.drop("label").to_dict()
    
    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
    result = response.json()
    
    predicted = result["prediction"]
    risk = result["risk_level"]
    match = "✅" if actual == predicted else "❌"
    
    print(f"{i+1:<4} {actual:<10} {predicted:<12} {risk:<8} {match}")

print("\nDone!")
