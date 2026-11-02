"""Train and persist the RetailMind customer churn prediction pipeline."""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .data_preprocessing import (
        clean_data,
        inspect_data,
        load_data,
        split_features_target,
    )
except ImportError:
    from data_preprocessing import (
        clean_data,
        inspect_data,
        load_data,
        split_features_target,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_customer_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_pipeline.joblib"
RANDOM_STATE = 42


def train_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Train the fixed baseline and return it with the held-out test data."""
    df = load_data(data_path)
    inspect_data(df)
    df = clean_data(df)
    X, y, numerical_columns, categorical_columns = split_features_target(df)

    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "Training requires both churn classes with at least two rows each "
            "to create a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    artifact_path = Path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    print(f"\nTrained pipeline saved to {artifact_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
