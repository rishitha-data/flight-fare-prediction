import os
import sys
import pandas as pd
import joblib
import numpy as np

from src.exceptions import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "flight_model_v1.pkl")
            self.model = None
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self):
        try:
            if self.model is None:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model not found at {self.model_path}")

                logger.info("Loading model from disk")
                self.model = joblib.load(self.model_path)

            return self.model

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            logger.info("Prediction started")

            model = self.load_model()

            # =========================
            # PREDICT (LOG SCALE)
            # =========================
            preds_log = model.predict(features)

            # =========================
            # CONVERT BACK (CRITICAL FIX)
            # =========================
            preds = np.expm1(preds_log)

            logger.info(f"Prediction completed: {preds}")

            return preds

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise CustomException(e, sys)