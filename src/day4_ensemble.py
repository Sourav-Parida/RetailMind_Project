"""Engineer customer features and compare Day 4 ensemble candidates."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_preprocessing import (
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    get_train_test_data,
    make_cv,
    make_preprocessor,
)
from evaluate import evaluate_classifier


DAY3_MODEL_PATH = MODEL_DIR / "churn_day3_tuned.joblib"


def add_customer_features(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Create the explainable Day 4 features used by the notebook."""
    engineered = input_frame.copy()

    if "tenure" in engineered.columns:
        tenure = pd.to_numeric(engineered["tenure"], errors="coerce")
        engineered["tenure_group"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        ).astype("object")

    if {"TotalCharges", "tenure"}.issubset(engineered.columns):
        total = pd.to_numeric(engineered["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(engineered["tenure"], errors="coerce").replace(
            0, np.nan
        )
        engineered["average_charge_per_month"] = total / tenure

    if "Contract" in engineered.columns:
        engineered["is_month_to_month"] = (
            engineered["Contract"] == "Month-to-month"
        ).astype(int)

    if "PaymentMethod" in engineered.columns:
        engineered["uses_electronic_check"] = (
            engineered["PaymentMethod"] == "Electronic check"
        ).astype(int)

    return engineered


def make_day4_pipeline(
    input_frame: pd.DataFrame,
    estimator,
) -> Pipeline:
    """Build a pipeline that keeps feature engineering and selection in each CV fold."""
    engineered_frame = add_customer_features(input_frame)
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(add_customer_features, validate=False),
            ),
            ("preprocess", make_preprocessor(engineered_frame)),
            ("feature_selection", SelectKBest(score_func=f_classif, k=30)),
            ("model", estimator),
        ]
    )


def _day4_candidates() -> dict[str, object]:
    candidates = {
        "Random Forest + SelectKBest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print(
            "XGBoost is not installed; only the Random Forest candidate will run."
        )
    else:
        candidates["XGBoost + SelectKBest"] = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return candidates


def run_day4(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    day3_champion=None,
    day3_champion_name: str | None = None,
    model_dir: str | Path = MODEL_DIR,
    report_dir: str | Path = REPORT_DIR,
) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    """Select the Day 4 champion by training CV recall and report test results."""
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    if day3_champion is None:
        if not DAY3_MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Day 3 model not found: {DAY3_MODEL_PATH}. "
                "Run Day 3 or the complete pipeline first."
            )
        day3_champion = joblib.load(DAY3_MODEL_PATH)

    rows = []
    fitted_models = {}
    cv = make_cv()
    for model_name, estimator in _day4_candidates().items():
        print(f"Cross-validating {model_name}...")
        candidate = make_day4_pipeline(X_train, estimator)
        recall_scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            scoring="recall",
            cv=cv,
            n_jobs=1,
        )
        candidate.fit(X_train, y_train)
        fitted_models[model_name] = candidate

        metrics = evaluate_classifier(
            f"Day 4 — {model_name}", candidate, X_test, y_test
        )
        metrics["CV_Recall_Mean"] = recall_scores.mean()
        metrics["CV_Recall_Std"] = recall_scores.std()
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values(
        "CV_Recall_Mean", ascending=False, ignore_index=True
    )
    champion_name = results.iloc[0]["Model"].replace("Day 4 — ", "", 1)
    champion = fitted_models[champion_name]

    if day3_champion_name is None:
        day3_champion_name = type(day3_champion.named_steps["model"]).__name__
        if day3_champion_name == "SVC":
            day3_champion_name = "SVM"
    day3_metrics = evaluate_classifier(
        f"Day 3 champion — {day3_champion_name}",
        day3_champion,
        X_test,
        y_test,
    )
    comparison = pd.DataFrame(
        [
            {
                **day3_metrics,
                "CV_Recall_Mean": np.nan,
                "CV_Recall_Std": np.nan,
            },
            *rows,
        ]
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "day4_ensemble_comparison.csv", index=False)
    comparison.to_csv(report_dir / "day4_vs_day3_comparison.csv", index=False)
    joblib.dump(champion, model_dir / "churn_day4_champion.joblib")

    print("Day 4 candidates (sorted by training CV recall):")
    print(results.to_string(index=False))
    print(f"Day 4 champion: {champion_name}")
    print("Day 4 versus Day 3 held-out comparison:")
    print(comparison.to_string(index=False))
    return results, champion, comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Day 4 feature-engineering and ensemble workflow."
    )
    parser.add_argument("--data", help="Optional path to the Telco CSV.")
    parser.add_argument(
        "--day3-model",
        type=Path,
        default=DAY3_MODEL_PATH,
        help="Day 3 model artifact used for held-out comparison.",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = get_train_test_data(args.data)
    if not args.day3_model.is_file():
        raise FileNotFoundError(
            f"Day 3 model not found: {args.day3_model}. "
            "Run `python -m src.hyperparameter_tuning` first."
        )
    day3_champion = joblib.load(args.day3_model)
    run_day4(
        X_train,
        y_train,
        X_test,
        y_test,
        day3_champion=day3_champion,
    )


if __name__ == "__main__":
    main()
