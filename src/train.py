"""Train and save the Day 1 logistic-regression baseline."""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data_preprocessing import (
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    get_train_test_data,
    make_model_pipeline,
)
from .evaluate import evaluate_classifier


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir=MODEL_DIR,
    report_dir=REPORT_DIR,
):
    """Fit the baseline pipeline, evaluate it, and save its artifact and metrics."""
    model = make_model_pipeline(
        X_train,
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X_train, y_train)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "churn_day1_baseline.joblib")

    metrics = evaluate_classifier(
        "Day 1 — Logistic Regression baseline", model, X_test, y_test
    )
    pd.DataFrame([metrics]).to_csv(
        report_dir / "day1_baseline_metrics.csv", index=False
    )
    print(f"Day 1 baseline saved to {model_dir / 'churn_day1_baseline.joblib'}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return model, metrics


def main() -> None:
    X_train, X_test, y_train, y_test = get_train_test_data()
    train_baseline(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
