"""Train a Keras ANN and benchmark it against the Day 4 classical champion."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from .data_preprocessing import (
        MODEL_DIR,
        RANDOM_STATE,
        REPORT_DIR,
        get_train_test_data,
    )
except ImportError:
    from data_preprocessing import (
        MODEL_DIR,
        RANDOM_STATE,
        REPORT_DIR,
        get_train_test_data,
    )


CLASSICAL_MODEL_PATH = MODEL_DIR / "churn_day4_champion.joblib"
ANN_MODEL_PATH = MODEL_DIR / "churn_day5_ann.keras"
ANN_METADATA_PATH = MODEL_DIR / "churn_day5_ann_preprocessing.json"
ANN_METRICS_PATH = REPORT_DIR / "day5_ann_vs_classical.csv"

_SOURCE_DIR = str(Path(__file__).resolve().parent)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)


def _prepare_inputs(
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_medians: dict[str, float],
) -> dict[str, np.ndarray]:
    """Convert raw tabular rows to model inputs using training-only medians."""
    inputs = {}
    if numeric_columns:
        numeric = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric_medians)
        inputs["numeric_features"] = numeric.to_numpy(dtype=np.float32)

    for column in categorical_columns:
        values = features[column].fillna("").astype(str).to_numpy(dtype=object)
        inputs[column] = values

    return inputs


def build_ann(
    X_train: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_medians: dict[str, float],
) -> tf.keras.Model:
    """Build a compact Keras ANN with train-fitted normalization and lookups."""
    training_inputs = _prepare_inputs(
        X_train, numeric_columns, categorical_columns, numeric_medians
    )
    model_inputs = {}
    encoded_features = []

    if numeric_columns:
        numeric_input = tf.keras.Input(
            shape=(len(numeric_columns),),
            name="numeric_features",
            dtype=tf.float32,
        )
        normalizer = tf.keras.layers.Normalization(
            axis=-1, name="numeric_normalization"
        )
        normalizer.adapt(training_inputs["numeric_features"])
        encoded_features.append(normalizer(numeric_input))
        model_inputs["numeric_features"] = numeric_input

    for column in categorical_columns:
        categorical_input = tf.keras.Input(
            shape=(),
            name=column,
            dtype=tf.string,
        )
        lookup = tf.keras.layers.StringLookup(
            output_mode="one_hot",
            name=f"{column}_lookup",
        )
        lookup.adapt(training_inputs[column])
        encoded_features.append(lookup(categorical_input))
        model_inputs[column] = categorical_input

    if not encoded_features:
        raise ValueError("The ANN requires at least one numeric or categorical feature.")

    combined = (
        encoded_features[0]
        if len(encoded_features) == 1
        else tf.keras.layers.Concatenate(name="combined_features")(encoded_features)
    )
    hidden = tf.keras.layers.Dense(32, activation="relu", name="hidden_1")(combined)
    hidden = tf.keras.layers.Dense(16, activation="relu", name="hidden_2")(hidden)
    churn_probability = tf.keras.layers.Dense(
        1, activation="sigmoid", name="churn_probability"
    )(hidden)

    model = tf.keras.Model(inputs=model_inputs, outputs=churn_probability)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _classification_metrics(
    model_name: str,
    target: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, str | float]:
    labels = (probabilities >= 0.5).astype(int)
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(target, labels),
        "Precision": precision_score(target, labels, zero_division=0),
        "Recall": recall_score(target, labels, zero_division=0),
        "F1": f1_score(target, labels, zero_division=0),
        "ROC_AUC": roc_auc_score(target, probabilities),
    }


def train_and_benchmark_ann(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classical_model=None,
    classical_model_path: str | Path = CLASSICAL_MODEL_PATH,
    model_path: str | Path = ANN_MODEL_PATH,
    metadata_path: str | Path = ANN_METADATA_PATH,
    metrics_path: str | Path = ANN_METRICS_PATH,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 8,
) -> tuple[tf.keras.Model, pd.DataFrame, tf.keras.callbacks.History]:
    """Train with early stopping and compare against the existing champion."""
    if classical_model is None:
        path = Path(classical_model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Classical champion not found: {path}. Run the Day 4 workflow first."
            )
        classical_model = joblib.load(path)

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    numeric_columns = X_fit.select_dtypes(include="number").columns.tolist()
    categorical_columns = X_fit.select_dtypes(exclude="number").columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("The ANN requires at least one model feature.")

    numeric_medians = {}
    if numeric_columns:
        numeric_medians = (
            X_fit[numeric_columns]
            .apply(pd.to_numeric, errors="coerce")
            .median()
            .fillna(0.0)
            .astype(float)
            .to_dict()
        )
    training_inputs = _prepare_inputs(
        X_fit, numeric_columns, categorical_columns, numeric_medians
    )
    validation_inputs = _prepare_inputs(
        X_validation, numeric_columns, categorical_columns, numeric_medians
    )
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = build_ann(
        X_fit, numeric_columns, categorical_columns, numeric_medians
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        training_inputs,
        y_fit.to_numpy(dtype=np.float32),
        validation_data=(
            validation_inputs,
            y_validation.to_numpy(dtype=np.float32),
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    metadata = {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_medians": numeric_medians,
        "threshold": 0.5,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    ann_inputs = _prepare_inputs(
        X_test, numeric_columns, categorical_columns, numeric_medians
    )
    ann_probabilities = model.predict(ann_inputs, verbose=0).ravel()
    classical_probabilities = classical_model.predict_proba(X_test)[:, 1]
    metrics = pd.DataFrame(
        [
            _classification_metrics(
                "Day 4 classical champion",
                y_test,
                classical_probabilities,
            ),
            _classification_metrics(
                "Day 5 Keras ANN",
                y_test,
                ann_probabilities,
            ),
        ]
    )
    metrics.to_csv(metrics_path, index=False)
    print(f"ANN saved to {model_path}")
    print(f"ANN preprocessing metadata saved to {metadata_path}")
    print("Held-out comparison (threshold = 0.5):")
    print(metrics.to_string(index=False))
    print(f"Early stopping completed after {len(history.epoch)} epoch(s).")
    return model, metrics, history


def predict_day5(
    customer_data: pd.DataFrame,
    model_path: str | Path = ANN_MODEL_PATH,
    metadata_path: str | Path = ANN_METADATA_PATH,
) -> pd.DataFrame:
    """Predict churn using the saved Day 5 ANN and its preprocessing metadata."""

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Day 5 ANN not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Day 5 preprocessing metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = customer_data.drop(
        columns=["customerID", "Churn"], errors="ignore"
    ).copy()
    inputs = _prepare_inputs(
        features,
        metadata["numeric_columns"],
        metadata["categorical_columns"],
        metadata["numeric_medians"],
    )
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(inputs, verbose=0).ravel()
    threshold = metadata.get("threshold", 0.5)
    return pd.DataFrame(
        {
            "predicted_churn": np.where(probabilities >= threshold, "Yes", "No"),
            "churn_probability": probabilities,
        },
        index=customer_data.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and benchmark the Day 5 Keras ANN."
    )
    parser.add_argument("--data", help="Optional path to the Telco CSV.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--classical-model",
        type=Path,
        default=CLASSICAL_MODEL_PATH,
        help="Day 4 classical champion artifact.",
    )
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = get_train_test_data(args.data)

    train_and_benchmark_ann(
        X_train, y_train, X_test, y_test,
        classical_model_path=args.classical_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()