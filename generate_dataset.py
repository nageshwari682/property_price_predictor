from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(path: str | Path) -> None:
    rng = np.random.default_rng(42)
    locations = ["Downtown", "Suburban", "Riverside", "Old Town", "Hillview", "Uptown", "Lakeview"]
    property_types = ["Apartment", "Villa", "Townhouse", "Condo"]

    n_rows = 2500
    data = {
        "location": rng.choice(locations, size=n_rows),
        "area_sqft": rng.normal(2100, 550, n_rows).round(0).astype(int),
        "bedrooms": rng.integers(1, 6, size=n_rows),
        "bathrooms": rng.integers(1, 4, size=n_rows),
        "stories": rng.integers(1, 4, size=n_rows),
        "age_years": rng.integers(0, 40, size=n_rows),
        "garage": rng.integers(0, 3, size=n_rows),
        "balcony": rng.integers(0, 2, size=n_rows),
        "basement": rng.integers(0, 2, size=n_rows),
        "nearby_school_score": rng.uniform(5, 10, n_rows).round(2),
        "crime_rate": rng.uniform(0.05, 1.5, n_rows).round(2),
        "property_type": rng.choice(property_types, size=n_rows),
    }

    df = pd.DataFrame(data)
    loc_prices = {
        "Downtown": 120000,
        "Suburban": 90000,
        "Riverside": 105000,
        "Old Town": 77000,
        "Hillview": 140000,
        "Uptown": 130000,
        "Lakeview": 150000,
    }
    type_prices = {"Apartment": 35000, "Villa": 90000, "Townhouse": 60000, "Condo": 42000}

    df["price"] = (
        df["area_sqft"] * 180
        + df["bedrooms"] * 26000
        + df["bathrooms"] * 32000
        + df["stories"] * 48000
        + df["garage"] * 28000
        + df["balcony"] * 11000
        + df["basement"] * 19000
        + df["nearby_school_score"] * 20000
        - df["age_years"] * 1200
        - df["crime_rate"] * 40000
        + df["location"].map(loc_prices)
        + df["property_type"].map(type_prices)
        + rng.normal(0, 25000, n_rows)
    ).round(2)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    generate_dataset("data/raw/housing_data.csv")
