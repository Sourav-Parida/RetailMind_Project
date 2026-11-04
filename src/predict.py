"""Load the Day 3 champion and predict churn for new customer records."""

import argparse
from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd

from data_preprocessing import DATA_DIR, MODEL_DIR, find_dataset


MODEL_PATH = MODEL_DIR / "churn_day4_champion.joblib"


def predict(
    customer_data: Mapping | pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
) -> pd.DataFrame:
    """Return churn labels and probabilities for one or more customer records."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run `python -m src.run_pipeline` first."
        )

    if isinstance(customer_data, pd.DataFrame):
        features = customer_data.copy()
    elif isinstance(customer_data, Mapping):
        features = pd.DataFrame([customer_data])
    else:
        raise TypeError("customer_data must be a mapping or pandas DataFrame.")

    features = features.drop(columns=["customerID", "Churn"], errors="ignore")
    if features.empty or features.shape[1] == 0:
        raise ValueError("Provide at least one customer feature column.")
    if "TotalCharges" in features.columns:
        features["TotalCharges"] = pd.to_numeric(
            features["TotalCharges"], errors="coerce"
        )

    model = joblib.load(path)
    labels = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "predicted_churn": ["Yes" if label == 1 else "No" for label in labels],
            "churn_probability": probabilities,
        },
        index=features.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Try the Day 4 model on the first row of the dataset."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()

    data_path = args.data if args.data is not None else find_dataset(DATA_DIR)
    sample = pd.read_csv(data_path, nrows=1)
    print(predict(sample, args.model).to_string(index=False))


if __name__ == "__main__":
    main()
