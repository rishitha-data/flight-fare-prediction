from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import get_logger

# =========================
# LOGGER
# =========================
logger = get_logger(__name__)

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="Flight Fare Prediction API",
    description="Predict flight ticket prices using ML model",
    version="1.0.0"
)

# =========================
# LOAD PIPELINE
# =========================
pipeline = PredictPipeline()


# =========================
# INPUT SCHEMA
# =========================
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: int = Field(ge=0, le=4)
    Duration_Minutes: int = Field(gt=0)
    Journey_Day: int = Field(ge=1, le=31)
    Journey_Month: int = Field(ge=1, le=12)
    Is_Weekend: int = Field(ge=0, le=1)
    Dep_Hour: int = Field(ge=0, le=23)
    Dep_Min: int = Field(ge=0, le=59)
    Arr_Hour: int = Field(ge=0, le=23)
    Arr_Min: int = Field(ge=0, le=59)
    Is_Peak_Departure: int = Field(ge=0, le=1)


# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def home():
    return {"message": "Flight Fare Prediction API is running 🚀"}


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict(input_data: FlightInput):
    try:
        logger.info("Prediction request received")

        # =========================
        # CLEAN INPUT (VERY IMPORTANT)
        # =========================
        data = input_data.dict()

        data["Airline"] = data["Airline"].strip()
        data["Source"] = data["Source"].strip()
        data["Destination"] = data["Destination"].strip()

        # =========================
        # CREATE DATAFRAME
        # =========================
        df = pd.DataFrame([data])

        # =========================
        # PREDICT
        # =========================
        prediction = pipeline.predict(df)

        if prediction is None or len(prediction) == 0:
            raise ValueError("Prediction failed")

        price = float(prediction[0])

        # =========================
        # FIX NEGATIVE VALUES (CRITICAL)
        # =========================
        if price < 0:
            logger.warning(f"Negative prediction detected: {price}")
            price = 0

        logger.info(f"Prediction successful: {price}")

        return {
            "predicted_price": int(price)
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check input data."
        )