"""Use a VAE to augment the rare catalog category, then evaluate Day 7.

Generate candidates first, inspect the saved contact sheet, and copy only
realistic JPEGs into the approved/ folder before running --evaluate.
"""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import tensorflow as tf

try:
    from .day6_cnn import (
        AUTOTUNE,
        CATALOG_MODEL_PATH,
        CLASS_NAMES_PATH,
        CONFUSION_MATRIX_PATH,
        DATA_DIR,
        SEED,
        load_catalog_records,
        split_catalog_records,
    )
    from .day7_transfer_learning import (
        TRANSFER_MODEL_PATH,
        make_transfer_dataset,
    )
except ImportError:  # pragma: no cover
    from day6_cnn import (
        AUTOTUNE,
        CATALOG_MODEL_PATH,
        CLASS_NAMES_PATH,
        CONFUSION_MATRIX_PATH,
        DATA_DIR,
        SEED,
        load_catalog_records,
        split_catalog_records,
    )
    from day7_transfer_learning import (
        TRANSFER_MODEL_PATH,
        make_transfer_dataset,
    )


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SYNTHETIC_DIR = DATA_DIR / "day11_synthetic_free_items"
CANDIDATE_DIR = SYNTHETIC_DIR / "candidates"
APPROVED_DIR = SYNTHETIC_DIR / "approved"
VAE_DIR = MODEL_DIR / "day11_free_items_vae"
DECODER_PATH = VAE_DIR / "decoder.keras"
AUGMENTED_MODEL_PATH = MODEL_DIR / "day11_day7_resnet50_augmented.keras"
SAMPLE_GRID_PATH = REPORT_DIR / "day11_vae_candidates.png"
QUALITY_REPORT_PATH = REPORT_DIR / "day11_vae_quality.csv"
VAE_HISTORY_PATH = REPORT_DIR / "day11_vae_training_history.csv"
METRICS_PATH = REPORT_DIR / "day11_per_class_comparison.csv"
SUMMARY_PATH = REPORT_DIR / "day11_augmentation_comparison.csv"
TARGET_CATEGORY = "Free Items"
VAE_ARTICLE_TYPE = "Free Gifts"
VAE_IMAGE_SIZE = (64, 64)
LATENT_DIMENSION = 32
DEFAULT_VAE_EPOCHS = 100
DEFAULT_CANDIDATE_COUNT = 64
CLASSIFIER_EPOCHS = 3


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=False):
        mean, log_variance = self.encoder(inputs, training=training)
        latent = mean + tf.exp(0.5 * log_variance) * tf.random.normal(
            tf.shape(mean)
        )
        return self.decoder(latent, training=training)

    def _compute_vae_losses(self, images, training):
        mean, log_variance = self.encoder(images, training=training)
        latent = mean + tf.exp(0.5 * log_variance) * tf.random.normal(
            tf.shape(mean)
        )
        reconstructions = self.decoder(latent, training=training)
        pixel_loss = tf.keras.losses.binary_crossentropy(
            images, reconstructions
        )
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(pixel_loss, axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + log_variance - tf.square(mean) - tf.exp(log_variance),
                axis=1,
            )
        )
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self._compute_vae_losses(
                images, training=True
            )
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        total_loss, reconstruction_loss, kl_loss = self._compute_vae_losses(
            images, training=False
        )
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def build_vae():
    encoder_input = tf.keras.Input(shape=(*VAE_IMAGE_SIZE, 3))
    encoded = tf.keras.layers.Conv2D(
        32, 3, strides=2, padding="same", activation="relu"
    )(encoder_input)
    encoded = tf.keras.layers.Conv2D(
        64, 3, strides=2, padding="same", activation="relu"
    )(encoded)
    encoded = tf.keras.layers.Conv2D(
        128, 3, strides=2, padding="same", activation="relu"
    )(encoded)
    encoded = tf.keras.layers.Flatten()(encoded)
    encoded = tf.keras.layers.Dense(128, activation="relu")(encoded)
    latent_mean = tf.keras.layers.Dense(LATENT_DIMENSION, name="latent_mean")(
        encoded
    )
    latent_log_variance = tf.keras.layers.Dense(
        LATENT_DIMENSION, name="latent_log_variance"
    )(encoded)
    encoder = tf.keras.Model(
        encoder_input, [latent_mean, latent_log_variance], name="vae_encoder"
    )

    decoder_input = tf.keras.Input(shape=(LATENT_DIMENSION,))
    decoded = tf.keras.layers.Dense(8 * 8 * 128, activation="relu")(
        decoder_input
    )
    decoded = tf.keras.layers.Reshape((8, 8, 128))(decoded)
    decoded = tf.keras.layers.Conv2DTranspose(
        128, 3, strides=2, padding="same", activation="relu"
    )(decoded)
    decoded = tf.keras.layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", activation="relu"
    )(decoded)
    decoded = tf.keras.layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", activation="relu"
    )(decoded)
    decoder_output = tf.keras.layers.Conv2D(
        3, 3, padding="same", activation="sigmoid"
    )(decoded)
    decoder = tf.keras.Model(decoder_input, decoder_output, name="vae_decoder")
    vae = VariationalAutoencoder(encoder, decoder, name="free_items_vae")
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return vae


def _read_confusion_support():
    if not CONFUSION_MATRIX_PATH.is_file():
        raise FileNotFoundError(
            "The Day 6 catalog confusion matrix is missing. Run "
            "python src/day6_cnn.py first."
        )
    with CONFUSION_MATRIX_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2:
        raise ValueError(f"Invalid confusion matrix: {CONFUSION_MATRIX_PATH}")
    class_names = rows[0][1:]
    support = {}
    recalls = {}
    for row in rows[1:]:
        if not row:
            continue
        counts = [int(value) for value in row[1:]]
        total = sum(counts)
        support[row[0]] = total
        class_index = class_names.index(row[0])
        recalls[row[0]] = counts[class_index] / total if total else 0.0
    return support, recalls


def identify_target_category(records):
    """Choose the least-supported modeled class with poor saved recall."""
    counts = {}
    for _, label in records:
        counts[label] = counts.get(label, 0) + 1
    support, recalls = _read_confusion_support()
    modeled_classes = set(support).intersection(counts)
    if TARGET_CATEGORY not in modeled_classes:
        raise ValueError(
            f"Expected {TARGET_CATEGORY!r} in the catalog confusion matrix."
        )
    target_count = counts[TARGET_CATEGORY]
    target_recall = recalls[TARGET_CATEGORY]
    print(
        "Rare category selected from Day 6 results: "
        f"{TARGET_CATEGORY} ({target_count} images; "
        f"{support[TARGET_CATEGORY]} test examples; "
        f"saved recall {target_recall:.3f})"
    )
    if target_count < 8:
        raise ValueError(
            f"Only {target_count} {TARGET_CATEGORY} images are available; "
            "there are too few to train a VAE."
        )
    return TARGET_CATEGORY, counts, support, recalls


def _load_article_types():
    csv_path = DATA_DIR / "styles.csv"
    if not csv_path.is_file():
        csv_path = DATA_DIR / "myntradataset" / "styles.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find styles.csv under {DATA_DIR}.")
    image_dir = csv_path.parent / "images"
    article_types = {}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"id", "articleType"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "styles.csv must include id and articleType to select "
                "the dominant Free Items subtype."
            )
        for row in reader:
            image_id = (row.get("id") or "").strip()
            image_path = image_dir / f"{image_id}.jpg"
            if image_id and image_path.is_file():
                article_types[str(image_path)] = (
                    row.get("articleType") or ""
                ).strip()
    return article_types


def _load_training_images(paths, batch_size=8):
    if not paths:
        raise ValueError("No training images were supplied to the VAE.")

    def decode(path):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, VAE_IMAGE_SIZE)
        return tf.cast(image, tf.float32) / 255.0

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    return (
        dataset.map(decode, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


def _save_candidate_grid(images, candidate_paths):
    columns = 8
    rows = math.ceil(len(images) / columns)
    padding_count = rows * columns - len(images)
    if padding_count:
        images = tf.concat(
            [
                images,
                tf.ones(
                    (padding_count, *VAE_IMAGE_SIZE, 3),
                    dtype=images.dtype,
                ),
            ],
            axis=0,
        )
    grid = tf.reshape(
        images, (rows, columns, *VAE_IMAGE_SIZE, 3)
    )
    grid = tf.transpose(grid, (0, 2, 1, 3, 4))
    grid = tf.reshape(
        grid, (rows * VAE_IMAGE_SIZE[0], columns * VAE_IMAGE_SIZE[1], 3)
    )
    SAMPLE_GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    png = tf.image.encode_png(
        tf.cast(tf.clip_by_value(grid, 0, 1) * 255, tf.uint8)
    )
    tf.io.write_file(str(SAMPLE_GRID_PATH), png)

    flattened = tf.reshape(images, (len(images), -1))
    pairwise_difference = tf.reduce_mean(
        tf.square(flattened[:, tf.newaxis, :] - flattened[tf.newaxis, :, :]),
        axis=2,
    )
    mask = tf.linalg.band_part(
        tf.ones_like(pairwise_difference, dtype=tf.bool), 0, -1
    ) & ~tf.eye(len(images), dtype=tf.bool)
    diversity_values = tf.boolean_mask(pairwise_difference, mask)
    mean_diversity = (
        float(tf.reduce_mean(diversity_values).numpy())
        if tf.size(diversity_values) > 0 else 0.0
    )
    with QUALITY_REPORT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category", "candidate_count", "sampling_method",
                "mean_pairwise_mse",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "category": TARGET_CATEGORY,
            "candidate_count": len(candidate_paths),
            "sampling_method": "small perturbations around training posteriors",
            "mean_pairwise_mse": mean_diversity,
        })


def generate_candidates(vae_epochs=DEFAULT_VAE_EPOCHS, candidate_count=DEFAULT_CANDIDATE_COUNT):
    if not CATALOG_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Train the Day 6 catalog model first. Expected "
            f"{CATALOG_MODEL_PATH.name} and {CLASS_NAMES_PATH.name}."
        )
    records = load_catalog_records()
    target_category, counts, _, _ = identify_target_category(records)
    train_data, val_data, _, class_names = split_catalog_records(records)
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        saved_class_names = json.load(file)
    if class_names != saved_class_names:
        raise ValueError(
            "Day 6 class labels differ from the current catalog split. "
            "Retrain the Day 6 catalog model first."
        )
    article_types = _load_article_types()
    category_id = class_names.index(target_category)
    rare_train_paths = [
        path
        for path, label in zip(*train_data)
        if label == category_id
        and article_types.get(path) == VAE_ARTICLE_TYPE
    ]
    rare_val_paths = [
        path
        for path, label in zip(*val_data)
        if label == category_id
        and article_types.get(path) == VAE_ARTICLE_TYPE
    ]
    if len(rare_train_paths) < 8 or not rare_val_paths:
        raise ValueError(
            f"Need at least 8 training and 1 validation images for "
            f"{target_category}/{VAE_ARTICLE_TYPE}; found "
            f"{len(rare_train_paths)} and {len(rare_val_paths)}."
        )
    print(
        f"Training VAE on {len(rare_train_paths)} {VAE_ARTICLE_TYPE} "
        f"training images; "
        f"holding out {len(rare_val_paths)} validation images."
    )

    tf.keras.utils.set_random_seed(SEED)
    vae = build_vae()
    history = vae.fit(
        _load_training_images(rare_train_paths),
        validation_data=_load_training_images(rare_val_paths),
        epochs=vae_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    VAE_DIR.mkdir(parents=True, exist_ok=True)
    vae.encoder.save(VAE_DIR / "encoder.keras")
    vae.decoder.save(DECODER_PATH)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_images = tf.concat(
        list(_load_training_images(rare_train_paths)), axis=0
    )
    means, log_variances = vae.encoder.predict(
        training_images, verbose=0
    )
    selected_indices = tf.random.uniform(
        (candidate_count,),
        minval=0,
        maxval=len(rare_train_paths),
        dtype=tf.int32,
        seed=SEED,
    )
    selected_means = tf.gather(means, selected_indices)
    selected_log_variances = tf.gather(log_variances, selected_indices)
    latent_samples = selected_means + 0.12 * tf.exp(
        0.5 * selected_log_variances
    ) * tf.random.normal(
        (candidate_count, LATENT_DIMENSION), seed=SEED + 1
    )
    generated = tf.clip_by_value(vae.decoder.predict(latent_samples), 0, 1)
    candidate_paths = []
    for index, image in enumerate(generated):
        path = CANDIDATE_DIR / f"candidate_{run_id}_{index:04d}.jpg"
        tf.keras.utils.save_img(path, image)
        candidate_paths.append(str(path))
    _save_candidate_grid(generated, candidate_paths)

    with VAE_HISTORY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch"] + list(history.history),
        )
        writer.writeheader()
        for index in range(len(history.epoch)):
            row = {"epoch": index + 1}
            row.update({
                name: values[index]
                for name, values in history.history.items()
            })
            writer.writerow(row)

    trained_epochs = len(history.epoch)
    print(f"VAE training epochs: {trained_epochs}/{vae_epochs}")
    print(f"Saved decoder: {DECODER_PATH}")
    print(f"Saved candidate grid for visual review: {SAMPLE_GRID_PATH}")
    print(f"Saved {len(candidate_paths)} candidates in {CANDIDATE_DIR}")
    print(
        f"Available original {target_category} images: {counts[target_category]}. "
        "Review the grid and copy only realistic JPEG candidates into "
        f"{APPROVED_DIR} before classifier evaluation."
    )
    return candidate_paths


def _per_class_results(model, dataset, true_labels, class_names):
    probabilities = model.predict(dataset, verbose=0)
    predictions = tf.argmax(probabilities, axis=1, output_type=tf.int32)
    confusion = tf.math.confusion_matrix(
        true_labels, predictions, num_classes=len(class_names)
    ).numpy()
    rows = []
    for index, name in enumerate(class_names):
        support = int(confusion[index].sum())
        correct = int(confusion[index, index])
        rows.append({
            "class_name": name,
            "support": support,
            "correct": correct,
            "accuracy": correct / support if support else 0.0,
        })
    return rows, confusion


def _write_confusion(path, confusion, class_names):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + class_names)
        for name, row in zip(class_names, confusion):
            writer.writerow([name] + row.tolist())


def evaluate_augmented_classifier(epochs=CLASSIFIER_EPOCHS):
    if not TRANSFER_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Train the Day 7 transfer-learning model first. Expected "
            f"{TRANSFER_MODEL_PATH.name} and {CLASS_NAMES_PATH.name}."
        )
    approved_paths = sorted(
        str(path) for path in APPROVED_DIR.glob("*.jpg")
        if path.is_file()
    )
    if not approved_paths:
        raise FileNotFoundError(
            "No approved synthetic JPEGs found. Inspect "
            f"{SAMPLE_GRID_PATH} and copy realistic candidates into "
            f"{APPROVED_DIR}."
        )

    records = load_catalog_records()
    train_data, val_data, test_data, class_names = split_catalog_records(
        records
    )
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        saved_class_names = json.load(file)
    if class_names != saved_class_names:
        raise ValueError(
            "Day 7 class labels differ from the current catalog split. "
            "Retrain Day 6 and Day 7 before augmentation evaluation."
        )
    category_id = class_names.index(TARGET_CATEGORY)
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    augmented_train_paths = train_paths + approved_paths
    augmented_train_labels = train_labels + [category_id] * len(approved_paths)

    baseline_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    baseline_rows, baseline_confusion = _per_class_results(
        baseline_model,
        make_transfer_dataset(test_paths, test_labels),
        test_labels,
        class_names,
    )
    baseline_loss, baseline_accuracy = baseline_model.evaluate(
        make_transfer_dataset(test_paths, test_labels), verbose=0
    )
    print(
        f"Day 7 baseline accuracy: {baseline_accuracy:.4f}; "
        f"{TARGET_CATEGORY} recall: "
        f"{baseline_rows[category_id]['accuracy']:.4f}"
    )

    augmented_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    augmented_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    augmented_model.fit(
        make_transfer_dataset(
            augmented_train_paths, augmented_train_labels, training=True
        ),
        validation_data=make_transfer_dataset(val_paths, val_labels),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    augmented_rows, augmented_confusion = _per_class_results(
        augmented_model,
        make_transfer_dataset(test_paths, test_labels),
        test_labels,
        class_names,
    )
    augmented_loss, augmented_accuracy = augmented_model.evaluate(
        make_transfer_dataset(test_paths, test_labels), verbose=0
    )
    augmented_model.save(AUGMENTED_MODEL_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model", "class_name", "support", "correct", "class_accuracy",
            ],
        )
        writer.writeheader()
        for model_name, rows in (
            ("Day 7 ResNet50 baseline", baseline_rows),
            ("Day 11 VAE-augmented ResNet50", augmented_rows),
        ):
            for row in rows:
                writer.writerow({
                    "model": model_name,
                    "class_name": row["class_name"],
                    "support": row["support"],
                    "correct": row["correct"],
                    "class_accuracy": row["accuracy"],
                })
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model", "test_loss", "test_accuracy",
                    "target_category_recall", "approved_synthetic_images",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "model": "Day 7 ResNet50 baseline",
            "test_loss": baseline_loss,
            "test_accuracy": baseline_accuracy,
            "target_category_recall": baseline_rows[category_id]["accuracy"],
            "approved_synthetic_images": 0,
        })
        writer.writerow({
            "model": "Day 11 VAE-augmented ResNet50",
            "test_loss": augmented_loss,
            "test_accuracy": augmented_accuracy,
            "target_category_recall": augmented_rows[category_id]["accuracy"],
            "approved_synthetic_images": len(approved_paths),
        })
    _write_confusion(
        REPORT_DIR / "day11_baseline_confusion_matrix.csv",
        baseline_confusion,
        class_names,
    )
    _write_confusion(
        REPORT_DIR / "day11_augmented_confusion_matrix.csv",
        augmented_confusion,
        class_names,
    )
    print(
        f"VAE-augmented accuracy: {augmented_accuracy:.4f}; "
        f"{TARGET_CATEGORY} recall: "
        f"{augmented_rows[category_id]['accuracy']:.4f}"
    )
    print(f"Saved augmented classifier: {AUGMENTED_MODEL_PATH}")
    print(f"Saved per-class metrics: {METRICS_PATH}")
    print(f"Saved comparison: {SUMMARY_PATH}")
    return baseline_rows, augmented_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Train the VAE and generate candidates for visual inspection.",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Fine-tune Day 7 using only manually approved candidates.",
    )
    parser.add_argument("--vae-epochs", type=int, default=DEFAULT_VAE_EPOCHS)
    parser.add_argument(
        "--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT
    )
    parser.add_argument(
        "--classifier-epochs", type=int, default=CLASSIFIER_EPOCHS
    )
    args = parser.parse_args()
    if args.vae_epochs < 1 or args.candidate_count < 1:
        parser.error("--vae-epochs and --candidate-count must be positive.")
    if args.classifier_epochs < 1:
        parser.error("--classifier-epochs must be positive.")
    if args.generate:
        generate_candidates(args.vae_epochs, args.candidate_count)
    else:
        evaluate_augmented_classifier(args.classifier_epochs)


if __name__ == "__main__":
    main()
