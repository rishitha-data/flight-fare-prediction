import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.preprocessing import preprocess_data


class TrainPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "flight_model_v1.pkl")
            self.metrics_path = os.path.join("artifacts", "metrics.json")
            self.test_size = 0.2
            self.random_state = 42
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, data_path: str):
        try:
            logging.info("Starting training pipeline")

            # ================================
            # LOAD DATA
            # ================================
            if not os.path.exists(data_path):
                raise FileNotFoundError("Dataset not found")

            df = pd.read_csv(data_path)
            logging.info(f"Dataset loaded with shape: {df.shape}")

            # ================================
            # PREPROCESS
            # ================================
            df = preprocess_data(df)

            X = df.drop("Price", axis=1)
            y = df["Price"]

            # ================================
            # TRAIN TEST SPLIT
            # ================================
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

            # ================================
            # MODEL
            # ================================
            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            logging.info("Training model")
            model.fit(X_train, y_train)

            # ================================
            # EVALUATION
            # ================================
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            logging.info(f"R2 Score: {r2}")
            logging.info(f"MAE: {mae}")

            # ================================
            # SAVE MODEL
            # ================================
            os.makedirs("artifacts", exist_ok=True)

            joblib.dump(model, self.model_path)

            # Save metrics
            metrics = {
                "r2": round(r2, 4),
                "mae": round(mae, 2)
            }

            import json
            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f)

            logging.info("Training pipeline completed successfully")

        except Exception as e:
            raise CustomException(e, sys)