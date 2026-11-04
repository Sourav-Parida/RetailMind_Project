"""Run the complete RetailMind churn workflow through Day 3."""

import argparse

from data_preprocessing import get_train_test_data
from day4_ensemble import run_day4
from hyperparameter_tuning import tune_models
from model_comparison import compare_models
from train import train_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the RetailMind churn project through Days 1-4."
    )
    parser.add_argument("--data", help="Optional path to the Telco CSV.")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = get_train_test_data(args.data)
    print(f"Train rows: {len(X_train)}; test rows: {len(X_test)}")
    print("\n=== Day 1: baseline ===")
    train_baseline(X_train, y_train, X_test, y_test)

    print("\n=== Day 2: candidate comparison ===")
    compare_models(X_train, y_train, X_test, y_test)
    
    print("\n=== Day 3: tuning and learning curve ===")
    day3_results, day3_champion, _, _ = tune_models(
        X_train, y_train, X_test, y_test
    )
    
    print("\n=== Day 4: feature engineering and ensemble comparison ===")
    run_day4(
        X_train,
        y_train,
        X_test,
        y_test,
        day3_champion=day3_champion,
        day3_champion_name=day3_results.iloc[0]["Model"],
    )
    print("\nCompleted Days 1-4.")


if __name__ == "__main__":
    main()
