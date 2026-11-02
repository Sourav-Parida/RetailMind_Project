"""Compare churn classifiers and persist the highest-recall champion."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from .data_preprocessing import clean_data, load_data, split_features_target
    from .train import (
        DATA_PATH,
        MODEL_PATH,
        RANDOM_STATE,
        build_preprocessor,
        split_data,
    )
except ImportError:
    from data_preprocessing import clean_data, load_data, split_features_target
    from train import (
        DATA_PATH,
        MODEL_PATH,
        RANDOM_STATE,
        build_preprocessor,
        split_data,
    )


MODEL_DIR = MODEL_PATH.parent


def _build_models() -> dict[str, tuple[str, object]]:
    """Return the baseline and Day 2 classifiers with fixed settings."""
    return {
        "Logistic Regression": (
            "churn_pipeline.joblib",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
        "Decision Tree": (
            "decision_tree.joblib",
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=12,
                random_state=RANDOM_STATE,
            ),
        ),
        "Random Forest": (
            "random_forest.joblib",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=200,
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "SVM": (
            "svm.joblib",
            SVC(
                class_weight="balanced",
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def _evaluate_model(model: Pipeline, X_test, y_test) -> dict[str, float]:
    """Calculate the shared model-comparison metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def compare_models(
    data_path: str | Path = DATA_PATH,
    model_dir: str | Path = MODEL_DIR,
) -> tuple[dict[str, dict[str, float]], str]:
    """Train and compare four pipelines on the same stratified test split."""
    df = clean_data(load_data(data_path))
    X, y, numerical_columns, categorical_columns = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    artifact_dir = Path(model_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_model: dict[str, dict[str, float]] = {}
    pipelines: dict[str, Pipeline] = {}

    for model_name, (artifact_name, classifier) in _build_models().items():
        print(f"\nTraining {model_name}...")
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(numerical_columns, categorical_columns),
                ),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = _evaluate_model(pipeline, X_test, y_test)
        metrics_by_model[model_name] = metrics
        pipelines[model_name] = pipeline
        joblib.dump(pipeline, artifact_dir / artifact_name)

        print(f"\n{model_name} Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # Recall is the primary criterion; ROC-AUC and precision break ties.
    champion_name = max(
        metrics_by_model,
        key=lambda name: (
            metrics_by_model[name]["recall"],
            metrics_by_model[name]["roc_auc"],
            metrics_by_model[name]["precision"],
        ),
    )
    champion_path = artifact_dir / "champion_model.joblib"
    joblib.dump(pipelines[champion_name], champion_path)

    print("\n========== CHAMPION MODEL ==========")
    print(f"Selected Model: {champion_name}")
    print(f"Saved champion pipeline to {champion_path}")
    return metrics_by_model, champion_name


if __name__ == "__main__":
    compare_models()
