"""Load the persisted pipeline and predict churn for a customer."""

from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from .train import MODEL_PATH
except ImportError:
    from train import MODEL_PATH


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    """Load the saved scikit-learn pipeline."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def predict(
    customer_data: Mapping[str, object],
    model_path: str | Path = MODEL_PATH,
) -> dict[str, object]:
    """Return a churn label and probability for one feature dictionary."""
    model = load_model(model_path)
    customer_df = pd.DataFrame([dict(customer_data)])

    expected_columns = list(model.feature_names_in_)
    missing_columns = sorted(set(expected_columns) - set(customer_df.columns))
    extra_columns = sorted(set(customer_df.columns) - set(expected_columns))
    if missing_columns or extra_columns:
        problems = []
        if missing_columns:
            problems.append(f"missing features: {missing_columns}")
        if extra_columns:
            problems.append(f"unexpected features: {extra_columns}")
        raise ValueError("Customer data has " + "; ".join(problems))

    customer_df = customer_df.loc[:, expected_columns]
    if "TotalCharges" in customer_df.columns:
        customer_df["TotalCharges"] = pd.to_numeric(
            customer_df["TotalCharges"], errors="coerce"
        )
    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0, 1])
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(probability, 4),
    }


if __name__ == "__main__":
    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
    print("RetailMind Prediction:\n")
    print(predict(example_customer))
