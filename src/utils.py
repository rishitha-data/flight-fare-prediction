import os
import sys
import json
import joblib

from src.exceptions import CustomException
from src.logger import logging


# ================================
# SAVE OBJECT (MODEL / PREPROCESSOR)
# ================================
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        joblib.dump(obj, file_path)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


# ================================
# LOAD OBJECT
# ================================
def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        obj = joblib.load(file_path)
        logging.info(f"Object loaded from {file_path}")

        return obj

    except Exception as e:
        raise CustomException(e, sys)


# ================================
# SAVE JSON (METRICS)
# ================================
def save_json(file_path, data: dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        logging.info(f"JSON saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


# ================================
# LOAD JSON
# ================================
def load_json(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "r") as f:
            data = json.load(f)

        logging.info(f"JSON loaded from {file_path}")

        return data

    except Exception as e:
        raise CustomException(e, sys)


# ================================
# VALIDATE DATAFRAME
# ================================
def validate_dataframe(df, required_columns):
    try:
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        logging.info("Data validation successful")

        return True

    except Exception as e:
        raise CustomException(e, sys)