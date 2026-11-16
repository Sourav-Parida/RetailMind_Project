"""Train a small 7-day Seq2Seq demand forecaster with attention.

Train Day 9 first, then run:
    python src/day10_seq2seq_attention.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

if __package__:
    from .day9_demand_forecasting import (
        LOOKBACK_DAYS,
        load_sales_frame,
        prepare_series,
        resolve_dataset_path,
    )
else:
    from day9_demand_forecasting import (
        LOOKBACK_DAYS,
        load_sales_frame,
        prepare_series,
        resolve_dataset_path,
    )


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "day10_seq2seq_attention.keras"
METRICS_PATH = ROOT / "reports" / "day10_forecast_metrics.csv"
PREDICTIONS_PATH = ROOT / "reports" / "day10_test_predictions.csv"
FORECAST_PATH = ROOT / "reports" / "day10_next_7_days.csv"
ATTENTION_PATH = ROOT / "reports" / "day10_attention.svg"
ATTENTION_CSV_PATH = ROOT / "reports" / "day10_attention_weights.csv"

HORIZON = 7
MAX_SERIES = 24
BATCH_SIZE = 256
EPOCHS = 5
SEED = 42


def build_model(lookback=LOOKBACK_DAYS, horizon=HORIZON):
    """Build a small encoder-decoder with attention over the history."""
    inputs = tf.keras.Input(shape=(lookback, 1))
    encoded, hidden, cell = tf.keras.layers.LSTM(
        16, return_sequences=True, return_state=True
    )(inputs)
    decoder_input = tf.keras.layers.RepeatVector(horizon)(hidden)
    decoded = tf.keras.layers.LSTM(
        16, return_sequences=True
    )(decoder_input, initial_state=[hidden, cell])
    scores = tf.keras.layers.Dot(axes=[2, 2])([decoded, encoded])
    weights = tf.keras.layers.Softmax(axis=-1, name="attention_weights")(scores)
    context = tf.keras.layers.Dot(axes=[2, 1])([weights, encoded])
    combined = tf.keras.layers.Concatenate()([decoded, context])
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(combined)
    model = tf.keras.Model(inputs, outputs, name="day10_seq2seq_attention")
    attention_model = tf.keras.Model(inputs, weights)
    model.compile(optimizer="adam", loss="mse")
    return model, attention_model


def select_series(series_list):
    """Use a spread of series to keep training short."""
    if len(series_list) <= MAX_SERIES:
        return series_list
    indices = [
        round(index * (len(series_list) - 1) / (MAX_SERIES - 1))
        for index in range(MAX_SERIES)
    ]
    return [series_list[index] for index in indices]


def make_examples(series_list, segment, horizon=HORIZON):
    """Make fixed-size history and future windows for one chronological split."""
    inputs, targets, records = [], [], []
    for series in series_list:
        if segment == "train":
            start, end = LOOKBACK_DAYS, series["validation_start"] - horizon + 1
        elif segment == "validation":
            start, end = series["validation_start"], series["test_start"] - horizon + 1
        elif segment == "test":
            start, end = series["test_start"], len(series["values"]) - horizon + 1
        else:
            raise ValueError(f"Unknown data split: {segment}")

        for target_start in range(start, end):
            inputs.append(
                series["scaled_values"][target_start - LOOKBACK_DAYS:target_start]
            )
            targets.append(series["scaled_values"][target_start:target_start + horizon])
            records.append((series, target_start))

    if not inputs:
        raise ValueError(f"No {segment} windows are available for this dataset.")
    x = np.asarray(inputs, dtype="float32")[..., np.newaxis]
    y = np.asarray(targets, dtype="float32")[..., np.newaxis]
    return x, y, records


def save_attention_svg(weights, path):
    """Save attention weights as a small SVG heatmap without extra libraries."""
    rows, columns = weights.shape
    cell = 18
    left, top = 70, 45
    rects = []
    for row in range(rows):
        for column in range(columns):
            intensity = min(255, int(float(weights[row, column]) * 255 * columns))
            color = f"#{255:02x}{255-intensity:02x}{255-intensity:02x}"
            rects.append(
                f'<rect x="{left + column * cell}" y="{top + row * cell}" '
                f'width="{cell}" height="{cell}" fill="{color}"/>'
            )
    width = left + columns * cell + 10
    height = top + rows * cell + 35
    labels = [
        f'<text x="{left + column * cell}" y="{top - 10}" '
        f'font-size="10">{column + 1}</text>'
        for column in range(columns)
    ]
    labels += [
        f'<text x="10" y="{top + row * cell + 13}" font-size="11">'
        f'Day {row + 1}</text>'
        for row in range(rows)
    ]
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<text x="10" y="20" font-size="14">Attention: forecast day to history day</text>'
        + "".join(labels + rects)
        + "</svg>",
        encoding="utf-8",
    )


def run(dataset_path=None):
    tf.keras.utils.set_random_seed(SEED)
    csv_path = resolve_dataset_path(dataset_path)
    frame, date_column, sales_column, group_columns = load_sales_frame(csv_path)
    series_list = prepare_series(
        frame, date_column, sales_column, group_columns, LOOKBACK_DAYS
    )
    series_list = select_series(series_list)
    print(f"Dataset: {csv_path.name} | training series: {len(series_list)}")

    x_train, y_train, _ = make_examples(series_list, "train")
    x_valid, y_valid, _ = make_examples(series_list, "validation")
    x_test, y_test, test_records = make_examples(series_list, "test")
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.shuffle(min(len(x_train), 10000), seed=SEED)
    train_data = train_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_data = valid_data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model, attention_model = build_model()
    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    day9_path = ROOT / "models" / "day9_demand_forecaster.keras"
    if not day9_path.is_file():
        raise FileNotFoundError(
            f"Train Day 9 first; its saved model is required for comparison: {day9_path}"
        )
    day9_model = tf.keras.models.load_model(day9_path)
    if day9_model.input_shape[1:] != (LOOKBACK_DAYS, 1):
        raise ValueError("The saved Day 9 model does not accept 28 daily values.")

    scaled_predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)[..., 0]
    baseline_scaled = day9_model.predict(
        x_test, batch_size=BATCH_SIZE, verbose=0
    ).reshape(-1)
    actual, predicted, baseline, prediction_rows = [], [], [], []
    for index, (series, target_start) in enumerate(test_records):
        actual_row = y_test[index, :, 0] * series["std"] + series["mean"]
        predicted_row = scaled_predictions[index] * series["std"] + series["mean"]
        predicted_row = np.maximum(0, predicted_row)
        baseline_value = max(
            0.0, float(baseline_scaled[index] * series["std"] + series["mean"])
        )
        actual.extend(actual_row.tolist())
        predicted.extend(predicted_row.tolist())
        baseline.append((float(actual_row[0]), baseline_value))
        for step, value in enumerate(predicted_row):
            row = {
                "forecast_date": str(series["dates"][target_start + step].date()),
                "horizon_day": step + 1,
                "actual_sales": float(actual_row[step]),
                "predicted_sales": float(value),
            }
            row.update(series["identifiers"])
            prediction_rows.append(row)

    day10_first_errors = [
        abs(float(y_test[i, 0, 0] * record[0]["std"] + record[0]["mean"])
            - float(scaled_predictions[i, 0] * record[0]["std"] + record[0]["mean"]))
        for i, record in enumerate(test_records)
    ]
    day9_first_errors = [abs(actual_value - forecast) for actual_value, forecast in baseline]
    errors = np.asarray(actual) - np.asarray(predicted)
    metrics = pd.DataFrame([{
        "dataset": csv_path.parent.name,
        "series_count": len(series_list),
        "test_windows": len(test_records),
        "horizon_days": HORIZON,
        "day9_one_day_mae": float(np.mean(day9_first_errors)),
        "day10_one_day_mae": float(np.mean(day10_first_errors)),
        "day10_all_horizons_mae": float(np.mean(np.abs(errors))),
        "day10_all_horizons_rmse": float(np.sqrt(np.mean(errors ** 2))),
    }])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    metrics.to_csv(METRICS_PATH, index=False)
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False)

    last_windows = np.asarray(
        [series["scaled_values"][-LOOKBACK_DAYS:] for series in series_list],
        dtype="float32",
    )[..., np.newaxis]
    next_values = model.predict(last_windows, verbose=0)[..., 0]
    next_rows = []
    for series, values in zip(series_list, next_values):
        for step, scaled_value in enumerate(values, start=1):
            row = {
                "forecast_date": str(
                    (series["dates"][-1] + pd.Timedelta(days=step)).date()
                ),
                "horizon_day": step,
                "forecast_sales": max(
                    0.0, float(scaled_value * series["std"] + series["mean"])
                ),
            }
            row.update(series["identifiers"])
            next_rows.append(row)
    pd.DataFrame(next_rows).to_csv(FORECAST_PATH, index=False)

    attention = attention_model.predict(x_test[:1], verbose=0)[0]
    weight_rows = [
        {"forecast_horizon_day": step + 1, "history_day": day + 1, "weight": float(weight)}
        for step, row in enumerate(attention)
        for day, weight in enumerate(row)
    ]
    pd.DataFrame(weight_rows).to_csv(ATTENTION_CSV_PATH, index=False)
    save_attention_svg(attention, ATTENTION_PATH)
    print(metrics.to_string(index=False))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved forecasts and attention files under {METRICS_PATH.parent}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to a supported sales train.csv")
    run(parser.parse_args().data)