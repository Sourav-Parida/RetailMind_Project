"""Loading, cleaning, and splitting the Telco Customer Churn data."""

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset from a CSV file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Telco Customer Churn CSV "
            "and save it as data/telco_customer_churn.csv."
        )

    df = pd.read_csv(path)
    print(f"Dataset loaded successfully from {path}!")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print useful information about the dataset before training."""
    print("\nDataset Shape:", df.shape)
    print("\nDataset Information:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())

    if "Churn" in df.columns:
        print("\nTarget Distribution:")
        print(df["Churn"].value_counts(dropna=False))
        print("\nTarget Percentage:")
        print(df["Churn"].value_counts(normalize=True, dropna=False) * 100)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers, coerce charge values, and discard missing targets."""
    if "Churn" not in df.columns:
        raise ValueError("Dataset must include the 'Churn' target column.")

    cleaned = df.copy()
    cleaned.drop(columns=["customerID"], errors="ignore", inplace=True)

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )

    cleaned.dropna(subset=["Churn"], inplace=True)
    return cleaned


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Separate features and binary target, returning column type lists."""
    if "Churn" not in df.columns:
        raise ValueError("Dataset must include the 'Churn' target column.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        invalid_targets = df.loc[y.isna(), "Churn"].unique().tolist()
        raise ValueError(f"Unexpected values in 'Churn': {invalid_targets}")
    if X.empty or X.shape[1] == 0:
        raise ValueError("Dataset must include at least one feature column.")

    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    return X, y.astype(int), numerical_columns, categorical_columns
