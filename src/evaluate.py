"""Evaluate the churn prediction pipeline on a stratified test split."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .train import train_model
except ImportError:
    from train import train_model


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Print and return accuracy, precision, recall, and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics


if __name__ == "__main__":
    trained_model, X_test, y_test = train_model()
    evaluate_model(trained_model, X_test, y_test)
