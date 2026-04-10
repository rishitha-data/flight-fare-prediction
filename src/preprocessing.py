import pandas as pd
import re
from typing import Tuple

from src.logger import get_logger  # ✅ FIXED

logger = get_logger(__name__)

# =============================
# REGEX (compiled once)
# =============================
HOUR_PATTERN = re.compile(r"(\d+)h")
MIN_PATTERN = re.compile(r"(\d+)m")
TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")


# =============================
# HELPERS
# =============================
def convert_duration(duration: str) -> int:
    if not isinstance(duration, str):
        return 0

    hours, minutes = 0, 0

    h = HOUR_PATTERN.search(duration)
    m = MIN_PATTERN.search(duration)

    if h:
        hours = int(h.group(1))
    if m:
        minutes = int(m.group(1))

    return hours * 60 + minutes


def extract_hour_min(time_str: str) -> Tuple[int, int]:
    if not isinstance(time_str, str):
        return 0, 0

    match = TIME_PATTERN.search(time_str)
    if match:
        return int(match.group(1)), int(match.group(2))

    return 0, 0


# =============================
# MAIN PREPROCESS FUNCTION
# =============================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Preprocessing started")

        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")

        df = df.copy()

        # =============================
        # DATE FEATURES
        # =============================
        if "Date_of_Journey" in df.columns:
            df["Date_of_Journey"] = pd.to_datetime(
                df["Date_of_Journey"], dayfirst=True, errors="coerce"
            )

            df["Journey_Day"] = df["Date_of_Journey"].dt.day.fillna(0).astype(int)
            df["Journey_Month"] = df["Date_of_Journey"].dt.month.fillna(0).astype(int)
            df["Is_Weekend"] = (
                df["Date_of_Journey"].dt.weekday >= 5
            ).fillna(False).astype(int)

        # =============================
        # TIME FEATURES
        # =============================
        if "Dep_Time" in df.columns:
            df[["Dep_Hour", "Dep_Min"]] = df["Dep_Time"].apply(
                lambda x: pd.Series(extract_hour_min(x))
            )

        if "Arrival_Time" in df.columns:
            df[["Arr_Hour", "Arr_Min"]] = df["Arrival_Time"].apply(
                lambda x: pd.Series(extract_hour_min(x))
            )

        if "Dep_Hour" in df.columns:
            df["Is_Peak_Departure"] = df["Dep_Hour"].apply(
                lambda x: 1 if (6 <= x <= 9 or 18 <= x <= 21) else 0
            )

        # =============================
        # DURATION
        # =============================
        if "Duration" in df.columns:
            df["Duration_Minutes"] = df["Duration"].apply(convert_duration)

        # =============================
        # TOTAL STOPS
        # =============================
        if "Total_Stops" in df.columns:
            df["Total_Stops"] = (
                df["Total_Stops"]
                .astype(str)
                .str.lower()
                .str.strip()
                .replace({
                    "non-stop": 0,
                    "non stop": 0,
                    "0 stop": 0,
                    "1 stop": 1,
                    "2 stops": 2,
                    "3 stops": 3,
                    "4 stops": 4,
                    "nan": None,
                    "": None
                })
            )

            df["Total_Stops"] = pd.to_numeric(
                df["Total_Stops"], errors="coerce"
            )

            median_val = df["Total_Stops"].median()
            if pd.isna(median_val):
                median_val = 0

            df["Total_Stops"] = df["Total_Stops"].fillna(median_val).astype(int)

        # =============================
        # DROP UNUSED COLUMNS
        # =============================
        drop_cols = [
            "Date_of_Journey",
            "Dep_Time",
            "Arrival_Time",
            "Duration",
            "Route",
            "Additional_Info"
        ]

        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        logger.info("Preprocessing completed successfully")

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise e   # ✅ FIXED