"""Shared evaluation metrics and standalone champion evaluation."""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_preprocessing import MODEL_DIR, get_train_test_data


def evaluate_classifier(
    model_name: str,
    fitted_model,
    features,
    target,
) -> dict[str, float | str]:
    """Calculate consistent binary classification metrics for churn."""
    predictions = fitted_model.predict(features)
    if hasattr(fitted_model, "predict_proba"):
        scores = fitted_model.predict_proba(features)[:, 1]
    else:
        scores = fitted_model.decision_function(features)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(target, scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Day 3 churn model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "churn_day3_tuned.joblib",
        help="Path to a saved model pipeline.",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {args.model}. Run the project pipeline first."
        )

    _, X_test, _, y_test = get_train_test_data()
    model = joblib.load(args.model)
    metrics = evaluate_classifier(args.model.stem, model, X_test, y_test)
    print("Held-out test metrics:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")
    predictions = model.predict(X_test)
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
