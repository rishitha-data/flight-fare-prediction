import pandas as pd
import json
import os
from typing import Dict

from src.config import TARGET, DATA_PATH, ARTIFACTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


# =========================
# OUTLIER DETECTION FUNCTION
# =========================
def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Dict:
    """
    Detect outliers using IQR method.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")

    # Ensure numeric column
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' must be numeric for outlier detection")

    # Drop NA safely
    df = df.dropna(subset=[column])

    if df.empty:
        raise ValueError("Column contains only NaN values")

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    total_records = len(df)
    outlier_count = len(outliers)

    return {
        "total_records": total_records,
        "outlier_count": outlier_count,
        "outlier_percentage": round(
            (outlier_count / total_records) * 100, 2
        ) if total_records > 0 else 0.0,
        "min_value": float(df[column].min()),
        "max_value": float(df[column].max()),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound)
    }


# =========================
# MAIN ANALYSIS FUNCTION
# =========================
def run_outlier_analysis() -> None:
    try:
        logger.info("Outlier analysis started")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        # =========================
        # LOAD DATA
        # =========================
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        if df.empty:
            raise ValueError("Dataset is empty")

        # =========================
        # RUN ANALYSIS
        # =========================
        results = detect_outliers_iqr(df, TARGET)

        # =========================
        # DISPLAY RESULTS
        # =========================
        print("\n📊 Outlier Analysis:")
        for key, value in results.items():
            print(f"{key}: {value}")

        # =========================
        # SAVE RESULTS
        # =========================
        output_path = os.path.join(
            ARTIFACTS_DIR, "outlier_report.json"
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Outlier report saved at {output_path}")
        logger.info("Outlier analysis completed successfully")

    except Exception as e:
        logger.error(f"Outlier analysis failed: {e}")
        print("❌ Error:", e)


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    run_outlier_analysis()