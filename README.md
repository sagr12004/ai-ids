
 # AI-Based Intrusion Detection System

A machine learning system that detects network intrusions using the NSL-KDD dataset. The system classifies network traffic as normal or an attack and assigns a risk level.

## Overview

This project implements a complete machine learning pipeline that processes network traffic data and predicts whether a connection is malicious. It is served as a REST API built with FastAPI.

## Dataset

The project uses the NSL-KDD dataset, which is an improved version of the KDD Cup 1999 dataset. It contains 41 features describing network connections and is widely used as a benchmark for intrusion detection research.

- Training set: 125,973 records
- Test set: 22,543 records

## Machine Learning Pipeline

The pipeline consists of the following stages:

1. Data Ingestion - Loads and labels the raw dataset
2. Data Validation - Checks for missing values, duplicates, and class distribution
3. Data Transformation - Encodes categorical features and scales numerical features
4. Model Training - Trains a Random Forest classifier
5. Model Evaluation - Evaluates performance on the test set
6. Prediction Pipeline - Handles incoming requests and returns predictions

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.44% |
| Precision | 66.23% |
| Recall    | 97.18% |
| F1 Score  | 78.77% |

The high recall score means the model catches nearly all attacks, which is the primary goal of an intrusion detection system.

## Tech Stack

- Python
- scikit-learn
- pandas
- FastAPI
- uvicorn
- joblib

## Project Structure
```
ai-ids/
├── data/               # Raw dataset files
├── artifacts/          # Trained model and preprocessing objects
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── model_evaluation.py
│   └── prediction_pipeline.py
├── app.py              # FastAPI application
├── test_samples.py     # Automated testing script
└── requirements.txt
```

## API Endpoints

| Method | Endpoint  | Description                        |
|--------|-----------|------------------------------------|
| GET    | /         | Health check                       |
| POST   | /predict  | Predict if network traffic is safe |

### Request Format

Send a POST request to /predict with all 41 network traffic features as a JSON body.

### Response Format
```json
{
  "prediction": "normal",
  "risk_level": "Low"
}
```

Prediction values: normal, attack
Risk level values: Low, High

## How to Run Locally

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies with pip install -r requirements.txt
4. Start the server with uvicorn app:app --reload
5. Open http://127.0.0.1:8000/docs in your browser

## Testing

Run the automated test script to validate predictions against real dataset samples:
```
python test_samples.py
```
