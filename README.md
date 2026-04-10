# ✈️ Flight Fare Prediction System

##  Live Demo

 API Base URL: https://flight-fare-prediction-hu2e.onrender.com  
 API Docs (Swagger): https://flight-fare-prediction-hu2e.onrender.com/docs  


##  Overview

An end-to-end **Machine Learning system** that predicts airline ticket prices for Indian domestic flights.

This project is built with a **production-ready architecture**, integrating:

* ML pipeline (feature engineering + model)
* FastAPI backend for real-time inference
* Streamlit frontend for interactive predictions
* Cloud deployment using Render

---

## Problem Statement

Airfare prices fluctuate based on multiple factors such as airline, route, duration, stops, and time.

This project aims to:

*  Analyze historical flight data
*  Train a predictive ML model
*  Serve predictions via API in real time

##  System Architecture

User (Streamlit UI) → FastAPI Backend → ML Pipeline → Prediction Output


##  Tech Stack

### Machine Learning

* Scikit-learn
* XGBoost
* Pandas, NumPy

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

### Deployment

* Render (Backend API)
* GitHub (Version Control)

### Utilities

* Joblib (model persistence)
* Logging module
* JSON (metrics storage)

---

## Features

### Machine Learning

* Advanced feature engineering (date, time, duration, stops)
* Pipeline-based preprocessing (scaling, encoding, imputation)
* XGBoost regression model
* Log transformation for improved performance
* Evaluation metrics: R², MAE, RMSE

### Backend (FastAPI)

* REST API for real-time predictions
* Input validation with Pydantic
* Health check endpoint
* Structured logging and error handling

### Frontend (Streamlit)

* Interactive UI
* Dynamic validation (e.g., source ≠ destination)
* Real-time predictions
* Trip summary view
* Estimated price range
* API latency indicator
* Reset functionality

### Engineering Practices

* Modular architecture (`src/`)
* Config-driven design
* Logging system
* Artifact management (model + metrics)

---

## Model Performance

| Metric   | Value     |
| -------- | --------- |
| R² Score | ~0.90+    |
| MAE      | Low       |
| RMSE     | Optimized |

---

## Project Structure

flight_fare_prediction/
│
├── artifacts/
├── data/
├── frontend/
│   └── app.py
├── src/
│   ├── pipeline/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── utils.py
│   ├── logger.py
│   └── exceptions.py
│
├── requirements.txt
├── render.yaml
└── README.md

---

## How to Run Locally

### 1️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Train Model

python -m src.train

### 4️⃣ Run Backend (FastAPI)

uvicorn src.api:app --reload

### 5️⃣ Run Frontend (Streamlit)

streamlit run frontend/app.py

---

## 🔌 API Endpoints

###  Health Check

GET /health

---

###  Predict Flight Price

POST /predict

#### Sample Input:

{
"Airline": "IndiGo",
"Source": "Delhi",
"Destination": "Cochin",
"Total_Stops": 1,
"Duration_Minutes": 150,
"Journey_Day": 10,
"Journey_Month": 3,
"Is_Weekend": 0,
"Dep_Hour": 10,
"Dep_Min": 0,
"Arr_Hour": 13,
"Arr_Min": 0,
"Is_Peak_Departure": 1
}


## Deployment

* Backend deployed on **Render**
* FastAPI server serving real-time predictions
* REST API accessible globally
* Swagger UI for testing endpoints

## Key Highlights

* ✔️ End-to-end ML system (data → model → API → UI)
* ✔️ Production-style architecture
* ✔️ Real-time prediction system
* ✔️ Cloud deployment
* ✔️ Industry best practices followed

## Future Improvements

* Add model versioning (MLflow)
* Docker containerization
* CI/CD pipeline integration
* User authentication
* Real-time data integration
