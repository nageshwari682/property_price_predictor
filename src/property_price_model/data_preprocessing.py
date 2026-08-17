from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def load_data(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Standardize column names
    cleaned.columns = [col.strip().lower().replace(" ", "_") for col in cleaned.columns]

    # Remove obviously invalid rows
    cleaned = cleaned.dropna(subset=["price"]).copy()
    numeric_columns = [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "stories",
        "age_years",
        "garage",
        "balcony",
        "basement",
        "nearby_school_score",
        "crime_rate",
        "price",
    ]
    for col in numeric_columns:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=numeric_columns)

    # Standardize categorical values
    if "location" in cleaned.columns:
        cleaned["location"] = cleaned["location"].astype(str).str.strip().str.title()
    if "property_type" in cleaned.columns:
        cleaned["property_type"] = cleaned["property_type"].astype(str).str.strip().str.title()

    # Price outlier handling for extreme values
    q1 = cleaned["price"].quantile(0.01)
    q99 = cleaned["price"].quantile(0.99)
    cleaned = cleaned[(cleaned["price"] >= q1) & (cleaned["price"] <= q99)]

    return cleaned.reset_index(drop=True)


def describe_dataset(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nSummary statistics:")
    print(df.describe(include="all").T)


def save_processed_data(df: pd.DataFrame, file_name: str = "processed_property_data.csv") -> str:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / file_name
    df.to_csv(output_path, index=False)
    return str(output_path)


def get_default_dataset_path() -> str:
    candidate_paths = [
        RAW_DATA_DIR / "housing_data.csv",
        RAW_DATA_DIR / "property_data.csv",
        RAW_DATA_DIR / "real_estate_data.csv",
        RAW_DATA_DIR / "dataset.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "No dataset found in data/raw/. Add a CSV file such as housing_data.csv or property_data.csv."
    )
