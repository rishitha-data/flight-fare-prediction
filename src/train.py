import pandas as pd
import joblib
import json
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from src.preprocessing import preprocess_data
from src.config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    METRICS_PATH,
    TARGET,
    DATA_PATH,
    ARTIFACTS_DIR,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    XGB_PARAMS
)
from src.logger import get_logger

logger = get_logger(__name__)


def train_model() -> None:
    try:
        logger.info("Training started")

        # ================================
        # SAFETY CHECKS
        # ================================
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

        # ================================
        # LOAD DATA
        # ================================
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            raise ValueError("Dataset is empty")

        df = preprocess_data(df)

        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' missing")

        # ================================
        # FEATURES & TARGET
        # ================================
        X = df.drop(TARGET, axis=1)

        # 🔥 Ensure target is valid
        y_raw = df[TARGET].clip(lower=0)

        # 🔥 LOG TRANSFORM
        y = np.log1p(y_raw)

        # ================================
        # SPLIT
        # ================================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # ================================
        # PIPELINES
        # ================================
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ])

        # ================================
        # MODEL
        # ================================
        model = XGBRegressor(
            **XGB_PARAMS,
            n_jobs=-1  # 🔥 performance improvement
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # ================================
        # TRAIN
        # ================================
        pipeline.fit(X_train, y_train)
        logger.info("Model training completed")

        # ================================
        # EVALUATE
        # ================================
        preds_log = pipeline.predict(X_test)

        # 🔥 BACK TRANSFORM
        preds = np.expm1(preds_log)
        y_test_actual = np.expm1(y_test)

        r2 = r2_score(y_test_actual, preds)
        mae = mean_absolute_error(y_test_actual, preds)
        rmse = np.sqrt(mean_squared_error(y_test_actual, preds))

        metrics = {
            "R2": round(r2, 3),
            "MAE": int(mae),
            "RMSE": int(rmse)
        }

        print("\n📊 Model Performance:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # ================================
        # SAVE MODEL
        # ================================
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"Model saved at {MODEL_PATH}")

        # ================================
        # SAVE METRICS
        # ================================
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("Metrics saved successfully")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print("❌ Error:", e)


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    train_model()