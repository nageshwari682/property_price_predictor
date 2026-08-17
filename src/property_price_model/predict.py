from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "property_price_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first by running: python src/property_price_model/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def prepare_input(sample: dict) -> pd.DataFrame:
    df = pd.DataFrame([sample])
    return df


def predict_property_price(sample: dict) -> float:
    model = load_model()
    features = prepare_input(sample)
    predicted = model.predict(features)[0]
    return float(predicted)
