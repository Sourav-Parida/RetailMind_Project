"""Compare the Day 2 candidate models and save the CV-selected champion."""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import (
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    get_train_test_data,
    make_cv,
    make_model_pipeline,
)
from .evaluate import evaluate_classifier


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir=MODEL_DIR,
    report_dir=REPORT_DIR,
) -> tuple[pd.DataFrame, object]:
    """Evaluate the Day 2 candidates and persist the highest-CV-recall model."""
    candidates = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    fitted_models = {}
    for name, estimator in candidates.items():
        pipeline = make_model_pipeline(X_train, estimator)
        cv_recall = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="recall",
            cv=make_cv(),
            n_jobs=1,
        ).mean()
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        metrics = evaluate_classifier(f"Day 2 — {name}", pipeline, X_test, y_test)
        metrics["CV_Recall"] = cv_recall
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values(
        "CV_Recall", ascending=False, ignore_index=True
    )
    champion_name = results.iloc[0]["Model"].replace("Day 2 — ", "", 1)
    champion = fitted_models[champion_name]

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "day2_model_comparison.csv", index=False)
    joblib.dump(champion, model_dir / "churn_day2_champion.joblib")
    print("Day 2 model comparison (sorted by training CV recall):")
    print(results.to_string(index=False))
    print(f"Day 2 champion: {champion_name}")
    return results, champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 2 model comparison.")
    parser.add_argument("--data", help="Optional path to the Telco CSV.")
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = get_train_test_data(args.data)
    compare_models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
