"""Run the Day 3 model searches and learning-curve analysis."""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import (
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    get_train_test_data,
    make_cv,
    make_model_pipeline,
)
from evaluate import evaluate_classifier


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir=MODEL_DIR,
    report_dir=REPORT_DIR,
):
    """Tune Day 3 candidates, save the CV champion, and write its learning curve."""
    search_spaces = {
        "Decision Tree": (
            DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            {
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [150, 250],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            {
                "model__C": [0.5, 1, 3],
                "model__gamma": ["scale", "auto"],
            },
        ),
    }

    best_models = {}
    rows = []
    cv = make_cv()
    
    for name, (estimator, parameter_grid) in search_spaces.items():
        print(f"Running Day 3 grid search for {name}...")
        search = GridSearchCV(
            estimator=make_model_pipeline(X_train, estimator),
            param_grid=parameter_grid,
            scoring="recall",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_models[name] = search.best_estimator_
        rows.append(
            {
                "Model": name,
                "Best_CV_Recall": search.best_score_,
                "Best_Parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    results = pd.DataFrame(rows).sort_values(
        "Best_CV_Recall", ascending=False, ignore_index=True
    )
    champion_name = results.iloc[0]["Model"]
    champion = best_models[champion_name]
    metrics = evaluate_classifier(
        f"Day 3 — Tuned {champion_name}", champion, X_test, y_test
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "day3_cv_tuning_results.csv", index=False)
    pd.DataFrame([metrics]).to_csv(
        report_dir / "day3_test_metrics.csv", index=False
    )
    joblib.dump(champion, model_dir / "churn_day3_tuned.joblib")

    sizes, train_scores, validation_scores = learning_curve(
        champion,
        X_train,
        y_train,
        cv=cv,
        scoring="recall",
        train_sizes=np.linspace(0.5, 1.0, 3),
        n_jobs=1,
    )
    curve = pd.DataFrame(
        {
            "Training_Size": sizes,
            "Mean_Training_Recall": train_scores.mean(axis=1),
            "Mean_CV_Recall": validation_scores.mean(axis=1),
        }
    )
    curve.to_csv(report_dir / "day3_learning_curve.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(
        sizes,
        train_scores.mean(axis=1),
        marker="o",
        label="Training recall",
    )
    plt.plot(
        sizes,
        validation_scores.mean(axis=1),
        marker="o",
        label="Cross-validation recall",
    )
    plt.xlabel("Number of training examples")
    plt.ylabel("Recall for Churn = Yes")
    plt.title("Day 3 — Learning curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(report_dir / "day3_learning_curve.png", dpi=140)
    plt.close()

    print("Day 3 tuning results (sorted by training CV recall):")
    print(results.to_string(index=False))
    print(f"Day 3 champion: {champion_name}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return results, champion, metrics, curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 3 tuning workflow.")
    parser.add_argument("--data", help="Optional path to the Telco CSV.")
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = get_train_test_data(args.data)
    tune_models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
