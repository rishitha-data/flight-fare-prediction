import pandas as pd
import joblib
import json
import os
import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from src.preprocessing import preprocess_data
from src.config import MODEL_PATH, METRICS_PATH, TARGET, DATA_PATH, ARTIFACTS_DIR
from src.logger import get_logger


# =========================
# LOGGER
# =========================
logger = get_logger(__name__)


def evaluate_model():
    try:
        logger.info("Evaluation started")

        # =========================
        # CHECK ARTIFACT DIR
        # =========================
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # =========================
        # LOAD MODEL
        # =========================
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")

        # =========================
        # LOAD DATA
        # =========================
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        df = preprocess_data(df)

        # =========================
        # SPLIT FEATURES
        # =========================
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in dataset")

        X = df.drop(TARGET, axis=1)
        y = df[TARGET]

        # =========================
        # PREDICTION
        # =========================
        preds = model.predict(X)

        # =========================
        # METRICS
        # =========================
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)

        # 🔥 FIXED RMSE (NO squared error issue)
        mse = mean_squared_error(y, preds)
        rmse = np.sqrt(mse)

        mape = mean_absolute_percentage_error(y, preds)

        metrics = {
            "R2": round(r2, 3),
            "MAE": int(mae),
            "RMSE": int(rmse),
            "MAPE (%)": round(mape * 100, 2)
        }

        # =========================
        # PRINT RESULTS
        # =========================
        print("\n📊 Model Evaluation:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # =========================
        # SAVE METRICS
        # =========================
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved at {METRICS_PATH}")
        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print("❌ Error:", e)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    evaluate_model()