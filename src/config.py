# config.py

import os

# =========================
# BASE PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "flight_data.csv")
)

ARTIFACTS_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "artifacts")
)

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================
# DATA CONFIG
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET = "Price"

# =========================
# FEATURE CONFIG
# =========================
CATEGORICAL_COLS = [
    "Airline",
    "Source",
    "Destination",
    "Total_Stops"
]

NUMERICAL_COLS = [
    "Duration_Minutes",
    "Journey_Day",
    "Journey_Month",
    "Dep_Hour",
    "Dep_Min",             # ✅ added
    "Arr_Hour",            # ✅ added
    "Arr_Min",             # ✅ added
    "Is_Weekend",          # ✅ added
    "Is_Peak_Departure"    # ✅ added
]

# =========================
# MODEL CONFIG
# =========================
MODEL_NAME = "xgboost"

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 700,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE
}

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "flight_model_v1.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
LOG_PATH = os.path.join(ARTIFACTS_DIR, "project.log")