from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def create_model_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = GradientBoostingRegressor(random_state=42, n_estimators=300, learning_rate=0.05, max_depth=3)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def split_features_target(df: pd.DataFrame, target_column: str = "price") -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target


def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "predictions": predictions,
    }


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def train_and_evaluate(df: pd.DataFrame, target_column: str = "price") -> tuple[Pipeline, dict]:
    X, y = split_features_target(df, target_column)
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    if not categorical_features:
        raise ValueError("No categorical features found. Add a categorical field such as location or property_type.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = create_model_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    metrics = evaluate_regression_model(model, X_test, y_test)

    return model, metrics
