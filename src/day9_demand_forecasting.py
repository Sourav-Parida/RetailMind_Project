"""Train an LSTM to forecast daily store or store-item demand.

Put a Rossmann ``train.csv`` in ``data/rossmann-store-sales/`` or a Store Item
Demand ``train.csv`` in ``data/store-item-demand-forecasting/``, then run:
    python src/day9_demand_forecasting.py

The script holds out the latest 20% of dates for testing, writes MAE/RMSE and
predictions to reports/, and saves the trained model under models/.
"""

import json
import math
from html import escape
from pathlib import Path

import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_CSV = None
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "day9_demand_forecaster.keras"
METADATA_PATH = MODEL_DIR / "day9_demand_forecaster_metadata.json"
METRICS_PATH = REPORT_DIR / "day9_demand_forecast_metrics.csv"
PREDICTIONS_PATH = REPORT_DIR / "day9_demand_test_predictions.csv"
NEXT_DAY_PATH = REPORT_DIR / "day9_demand_next_day_forecast.csv"
PLOT_PATH = REPORT_DIR / "day9_demand_forecast.svg"

LOOKBACK_DAYS = 28
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 256
EPOCHS = 20
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def resolve_dataset_path(dataset_path=None):
    """Find a supported Kaggle training CSV or validate a configured path."""
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Demand dataset CSV not found: {path}")
        return path

    candidates = [
        DATA_CSV,
        DATA_DIR / "rossmann-store-sales" / "train.csv",
        DATA_DIR / "store-item-demand-forecasting" / "train.csv",
        DATA_DIR / "train.csv",
    ]
    for candidate in candidates:
        if candidate is not None and Path(candidate).is_file():
            return Path(candidate)

    raise FileNotFoundError(
        "Could not find a demand dataset. Place Rossmann train.csv in "
        f"{DATA_DIR / 'rossmann-store-sales'} or Store Item Demand train.csv "
        f"in {DATA_DIR / 'store-item-demand-forecasting'}, or set DATA_CSV "
        "near the top of src/day9_demand_forecasting.py."
    )


def load_sales_frame(csv_path):
    """Read and validate the Rossmann or Store Item Demand CSV."""
    frame = pd.read_csv(csv_path)
    column_lookup = {str(column).strip().lower(): column for column in frame.columns}

    def find_column(candidates, required=True):
        for candidate in candidates:
            if candidate in column_lookup:
                return column_lookup[candidate]
        if required:
            expected = ", ".join(candidates)
            raise ValueError(
                f"{csv_path.name} must contain one of these columns: {expected}. "
                f"Found: {list(frame.columns)}"
            )
        return None

    date_column = find_column(("date",))
    sales_column = find_column(("sales",))
    group_columns = []
    for aliases in (("store", "store_id"), ("item", "item_id")):
        column = find_column(aliases, required=False)
        if column is not None:
            group_columns.append(column)
    if not any(str(column).strip().lower() in {"store", "store_id"}
               for column in group_columns):
        raise ValueError(
            "The demand data must contain a Store/store_id column to form "
            "separate store or store-item time series."
        )

    frame = frame[[date_column, *group_columns, sales_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[sales_column] = pd.to_numeric(frame[sales_column], errors="coerce")
    original_rows = len(frame)
    frame = frame.dropna(subset=[date_column, *group_columns, sales_column])
    if len(frame) < original_rows:
        print(
            f"Discarded {original_rows - len(frame)} rows with missing or "
            "invalid dates, series IDs, or sales values."
        )
    if frame.empty:
        raise ValueError("No valid dated sales rows were found in the CSV.")
    if (frame[sales_column] < 0).any():
        raise ValueError(
            f"{sales_column} contains negative values. Clean returns or "
            "invalid sales before training a demand forecaster."
        )

    return frame, date_column, sales_column, group_columns


def prepare_series(frame, date_column, sales_column, group_columns, lookback):
    """Aggregate and resample each store/store-item series to daily demand."""
    series_list = []
    skipped = 0
    for group_key, group in frame.groupby(
        group_columns, sort=False, dropna=False
    ):
        if len(group_columns) == 1:
            group_key = (group_key,)
        identifiers = dict(zip(group_columns, group_key))

        daily_sales = (
            group.set_index(date_column)[sales_column]
            .resample("D")
            .sum()
        )
        dates = pd.date_range(daily_sales.index.min(), daily_sales.index.max(), freq="D")
        daily_sales = daily_sales.reindex(dates, fill_value=0).astype("float32")
        values = daily_sales.to_numpy(dtype="float32")
        count = len(values)
        validation_start = int(count * (1 - TEST_FRACTION - VALIDATION_FRACTION))
        test_start = int(count * (1 - TEST_FRACTION))
        if validation_start <= lookback or test_start <= validation_start or count <= test_start:
            skipped += 1
            continue

        training_values = pd.Series(values[:validation_start])
        mean = float(training_values.mean())
        standard_deviation = float(training_values.std(ddof=0))
        if not math.isfinite(standard_deviation) or standard_deviation == 0:
            standard_deviation = 1.0
        scaled_values = (values - mean) / standard_deviation
        series_id = json.dumps(
            {str(key): str(value) for key, value in identifiers.items()},
            sort_keys=True,
        )
        series_list.append({
            "id": series_id,
            "identifiers": identifiers,
            "dates": dates,
            "values": values,
            "scaled_values": scaled_values.astype("float32"),
            "mean": mean,
            "std": standard_deviation,
            "validation_start": validation_start,
            "test_start": test_start,
        })

    if skipped:
        print(
            f"Skipping {skipped} series that are too short for a "
            f"{lookback}-day look-back and chronological validation/test splits."
        )
    if not series_list:
        raise ValueError(
            "No store/store-item series is long enough. Each retained series "
            f"needs more than {lookback} days plus validation and test dates."
        )
    return series_list


def count_windows(series_list, segment, lookback):
    count = 0
    for series in series_list:
        if segment == "train":
            start, end = lookback, series["validation_start"]
        elif segment == "validation":
            start, end = series["validation_start"], series["test_start"]
        elif segment == "test":
            start, end = series["test_start"], len(series["values"])
        else:
            raise ValueError(f"Unknown dataset segment: {segment}")
        count += max(0, end - start)
    return count


def iter_windows(series_list, segment, lookback):
    for series in series_list:
        if segment == "train":
            start, end = lookback, series["validation_start"]
        elif segment == "validation":
            start, end = series["validation_start"], series["test_start"]
        else:
            start, end = series["test_start"], len(series["values"])

        for target_index in range(start, end):
            window = series["scaled_values"][target_index - lookback:target_index]
            yield (
                tf.expand_dims(tf.convert_to_tensor(window, dtype=tf.float32), -1),
                tf.convert_to_tensor(
                    series["scaled_values"][target_index], dtype=tf.float32
                ),
            )


def make_dataset(series_list, segment, lookback, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: iter_windows(series_list, segment, lookback),
        output_signature=(
            tf.TensorSpec(shape=(lookback, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    if segment == "train":
        dataset = dataset.shuffle(
            buffer_size=20_000,
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def build_forecaster(lookback):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="retailmind_demand_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
    )
    return model


def _test_records(series_list):
    records = []
    for series in series_list:
        for target_index in range(series["test_start"], len(series["values"])):
            record = {
                "date": series["dates"][target_index],
                "actual_sales": float(series["values"][target_index]),
                "mean": series["mean"],
                "std": series["std"],
            }
            record.update(series["identifiers"])
            records.append(record)
    return records


def _save_svg_plot(predictions, path):
    representative = predictions.groupby(
        [column for column in predictions.columns
        if column not in {"date", "actual_sales", "predicted_sales"}],
        dropna=False,
        sort=False,
    ) if len(predictions.columns) > 3 else None
    if representative is not None:
        group_name, plotted = max(representative, key=lambda group: len(group[1]))
        if not isinstance(group_name, tuple):
            group_name = (group_name,)
        series_label = ", ".join(str(value) for value in group_name)
    else:
        plotted = predictions
        series_label = "Store"

    plotted = plotted.sort_values("date").tail(120)
    actual = plotted["actual_sales"].tolist()
    forecast = plotted["predicted_sales"].tolist()
    width, height = 900, 420
    left, right, top, bottom = 65, 20, 35, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max([0.0, *actual, *forecast])
    maximum = maximum * 1.1 if maximum > 0 else 1.0

    def point_coordinates(values):
        points = []
        denominator = max(1, len(values) - 1)
        for index, value in enumerate(values):
            x = left + index * plot_width / denominator
            y = top + plot_height * (1 - value / maximum)
            points.append(f"{x:.1f},{y:.1f}")
        return " ".join(points)

    start_date = pd.Timestamp(plotted["date"].iloc[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(plotted["date"].iloc[-1]).strftime("%Y-%m-%d")
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="65" y="22" font-family="sans-serif" font-size="16">'
            'Daily demand: actual vs predicted</text>'
        ),
        (
            f'<text x="65" y="40" font-family="sans-serif" font-size="11">'
            f'Example series: {escape(series_label)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{width-right}" y2="{top + plot_height}" stroke="#444"/>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#444"/>',
        (
            f'<polyline points="{point_coordinates(actual)}" fill="none" '
            'stroke="#1769aa" stroke-width="2"/>'
        ),
        (
            f'<polyline points="{point_coordinates(forecast)}" fill="none" '
            'stroke="#e87500" stroke-width="2"/>'
        ),
        (
            f'<text x="{left}" y="{height-18}" font-family="sans-serif" '
            f'font-size="11">{escape(start_date)}</text>'
        ),
        (
            f'<text x="{width-right-75}" y="{height-18}" '
            f'font-family="sans-serif" font-size="11">'
            f'{escape(end_date)}</text>'
        ),
        '<line x1="670" y1="22" x2="700" y2="22" stroke="#1769aa" stroke-width="3"/>',
        '<text x="706" y="26" font-family="sans-serif" font-size="12">Actual</text>',
        '<line x1="770" y1="22" x2="800" y2="22" stroke="#e87500" stroke-width="3"/>',
        '<text x="806" y="26" font-family="sans-serif" font-size="12">Predicted</text>',
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def run_forecasting(
    csv_path=None,
    model_path=MODEL_PATH,
    metadata_path=METADATA_PATH,
    report_dir=REPORT_DIR,
    lookback=LOOKBACK_DAYS,
    epochs=EPOCHS,
):
    """Train, evaluate, and save the demand forecaster and its reports."""
    tf.keras.utils.set_random_seed(SEED)
    csv_path = resolve_dataset_path(csv_path)
    frame, date_column, sales_column, group_columns = load_sales_frame(csv_path)
    series_list = prepare_series(
        frame, date_column, sales_column, group_columns, lookback
    )
    train_count = count_windows(series_list, "train", lookback)
    validation_count = count_windows(series_list, "validation", lookback)
    test_count = count_windows(series_list, "test", lookback)
    print(
        f"Dataset: {csv_path}\n"
        f"Series: {len(series_list)} | "
        f"Train windows: {train_count} | "
        f"Validation windows: {validation_count} | Test windows: {test_count}"
    )

    train_ds = make_dataset(series_list, "train", lookback, BATCH_SIZE)
    validation_ds = make_dataset(
        series_list, "validation", lookback, BATCH_SIZE
    )
    test_ds = make_dataset(series_list, "test", lookback, BATCH_SIZE)
    model = build_forecaster(lookback)
    model.summary()
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    report_dir = Path(report_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        steps_per_epoch=math.ceil(train_count / BATCH_SIZE),
        validation_steps=math.ceil(validation_count / BATCH_SIZE),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                save_best_only=True,
            ),
        ],
        verbose=1,
    )
    best_model = tf.keras.models.load_model(model_path)
    test_steps = math.ceil(test_count / BATCH_SIZE)
    test_loss = float(best_model.evaluate(
        test_ds, steps=test_steps, verbose=1
    ))
    test_predictions = best_model.predict(
        test_ds, steps=test_steps, verbose=1
    ).reshape(-1)
    target_records = _test_records(series_list)
    if len(target_records) != len(test_predictions):
        raise RuntimeError(
            "Prediction count does not match the chronological test windows."
        )

    for record, scaled_prediction in zip(target_records, test_predictions):
        record["predicted_sales"] = max(
            0.0, float(scaled_prediction) * record["std"] + record["mean"]
        )
    predictions = pd.DataFrame(target_records)
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    predictions = predictions.drop(columns=["mean", "std"])
    error = predictions["actual_sales"] - predictions["predicted_sales"]
    mae = float(error.abs().mean())
    rmse = float((error.pow(2).mean()) ** 0.5)
    predictions_path = report_dir / PREDICTIONS_PATH.name
    predictions.to_csv(predictions_path, index=False)

    forecast_windows = [
        series["scaled_values"][-lookback:] for series in series_list
    ]
    forecast_batch = tf.expand_dims(
        tf.convert_to_tensor(forecast_windows, dtype=tf.float32), -1
    )
    scaled_forecasts = best_model.predict(forecast_batch, verbose=0).reshape(-1)
    next_day_rows = []
    for series, scaled_forecast in zip(series_list, scaled_forecasts):
        row = {
            "forecast_date": (
                pd.Timestamp(series["dates"][-1]) + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d"),
            "forecast_sales": max(
                0.0, float(scaled_forecast) * series["std"] + series["mean"]
            ),
        }
        row.update(series["identifiers"])
        next_day_rows.append(row)
    next_day = pd.DataFrame(next_day_rows)
    next_day_path = report_dir / NEXT_DAY_PATH.name
    next_day.to_csv(next_day_path, index=False)

    metrics = pd.DataFrame([{
        "dataset": csv_path.parent.name or csv_path.stem,
        "series_count": len(series_list),
        "train_windows": train_count,
        "validation_windows": validation_count,
        "test_windows": test_count,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_loss_scaled_mse": test_loss,
        "lookback_days": lookback,
        "epochs_trained": len(history.epoch),
        "test_start": predictions["date"].min(),
        "test_end": predictions["date"].max(),
    }])
    metrics_path = report_dir / METRICS_PATH.name
    metrics.to_csv(metrics_path, index=False)
    plot_path = report_dir / PLOT_PATH.name
    _save_svg_plot(predictions, plot_path)

    metadata = {
        "dataset": csv_path.parent.name or csv_path.stem,
        "date_column": date_column,
        "sales_column": sales_column,
        "group_columns": group_columns,
        "frequency": "daily",
        "lookback_days": lookback,
        "series_scalers": {
            series["id"]: {"mean": series["mean"], "std": series["std"]}
            for series in series_list
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved next-day forecasts: {next_day_path}")
    print(f"Saved actual-vs-predicted plot: {plot_path}")
    print(f"Saved model metadata: {metadata_path}")
    return metrics


def forecast_next_demand(history, series_identifiers, model=None):
    """Forecast one series' next daily demand using the saved Day 9 model.

    ``series_identifiers`` maps the Store/Item columns used during training
    to their values. ``history`` contains that series' latest daily sales.
    """
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Forecaster metadata not found: {METADATA_PATH}. Train Day 9 first."
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Forecaster model not found: {MODEL_PATH}. Train Day 9 first."
            )
        model = tf.keras.models.load_model(MODEL_PATH)

    key_values = {}
    for column in metadata["group_columns"]:
        if column not in series_identifiers:
            raise ValueError(
                f"Missing series identifier {column!r}; provide all grouping "
                "columns used by the training dataset."
            )
        key_values[column] = str(series_identifiers[column])
    series_id = json.dumps(key_values, sort_keys=True)
    scaler = metadata["series_scalers"].get(series_id)
    if scaler is None:
        raise ValueError(
            f"No saved scaler for series {key_values}. The trained model "
            "supports the store/store-item series included in its training data."
        )

    sales = pd.to_numeric(pd.Series(history), errors="coerce")
    if sales.isna().any() or not all(math.isfinite(float(value)) for value in sales):
        raise ValueError("History must contain only finite numeric daily sales.")
    lookback = int(metadata["lookback_days"])
    if len(sales) < lookback:
        raise ValueError(
            f"Provide at least {lookback} days of sales history; received "
            f"{len(sales)}."
        )
    values = sales.iloc[-lookback:].to_numpy(dtype="float32")
    scaled = (values - scaler["mean"]) / scaler["std"]
    batch = tf.expand_dims(
        tf.convert_to_tensor([scaled], dtype=tf.float32), -1
    )
    scaled_prediction = float(model.predict(batch, verbose=0)[0][0])
    return max(0.0, scaled_prediction * scaler["std"] + scaler["mean"])


def main():
    run_forecasting()


if __name__ == "__main__":
    main()
