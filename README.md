# ✈️ Flight Fare Prediction System

##  Overview

An end-to-end **Machine Learning system** that predicts airline ticket prices for Indian domestic flights.
The project is designed with a **production-ready architecture**, integrating a trained ML model with a FastAPI backend and a Streamlit frontend for real-time predictions.


##  Problem Statement

Airfare prices fluctuate based on multiple factors such as airline, route, duration, stops, and time.
This project aims to **predict ticket prices accurately** using historical flight data and machine learning techniques.

---

##  System Architecture

```
Frontend (Streamlit UI)
        ↓
Backend (FastAPI API)
        ↓
ML Pipeline (Preprocessing + XGBoost Model)
```

---

##  Tech Stack

###  Core

* Python
* Pandas, NumPy

###  Machine Learning

* Scikit-learn
* XGBoost

###  Backend

* FastAPI
* Uvicorn

###  Frontend

* Streamlit

###  Utilities

* Joblib (model persistence)
* Logging module
* JSON (metrics tracking)

---

##  Features

###  Machine Learning

* Advanced feature engineering (date, time, duration, stops)
* Pipeline-based preprocessing (scaling, encoding, imputation)
* XGBoost regression model
* Log transformation for better accuracy
* Model evaluation with R², MAE, RMSE

###  Backend (FastAPI)

* REST API for predictions
* Input validation using Pydantic
* Health check endpoint
* Error handling and logging

###  Frontend (Streamlit)

* Interactive UI for inputs
* Real-time predictions
* Dynamic validation (e.g., source ≠ destination)
* Trip summary view
* Estimated price range
* API latency display
* Reset functionality

###  Engineering Practices

* Modular code structure (`src/`)
* Config-driven design
* Logging system
* Artifact management (model + metrics)

---

##  Model Performance

| Metric   | Value     |
| -------- | --------- |
| R² Score | ~0.90+    |
| MAE      | Low       |
| RMSE     | Optimized |

---

##  Project Structure

```
flight_fare_prediction/
│
├── artifacts/              # Model, metrics, logs
├── data/                   # Dataset
├── frontend/               # Streamlit app
│   └── app.py
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
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
└── README.md
```

##  How to Run

###  Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python -m src.train
```

### Run Backend (FastAPI)

```bash
uvicorn src.api:app --reload
```

### Run Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

---

##  API Endpoints

### ➤ Health Check

```
GET /health
```

### ➤ Predict Price

```
POST /predict
```

#### Sample Input:

```json
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
```

##  Key Highlights

* ✔️ End-to-end ML system (not just a model)
* ✔️ Production-style architecture
* ✔️ Real-time inference using API
* ✔️ Clean and intuitive UI
* ✔️ Industry best practices followed

## Future Improvements

* Add model versioning (MLflow)
* Deploy using Docker
* CI/CD pipeline integration
* Add user authentication
* Real-time data integration
Deployment fix trigger