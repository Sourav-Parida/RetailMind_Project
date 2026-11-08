"""Train CNNs for Fashion-MNIST and Fashion Product Images."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from .data_preprocessing import DATA_DIR, MODEL_DIR, RANDOM_STATE, REPORT_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_preprocessing import DATA_DIR, MODEL_DIR, RANDOM_STATE, REPORT_DIR


FASHION_MODEL = MODEL_DIR / "day6_fashion_mnist_cnn.keras"
CATALOG_MODEL = MODEL_DIR / "day6_product_catalog_cnn.keras"
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64


def build_cnn(input_shape, class_count):
    """Create the image classifier."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_indices(labels):
    """Split every class into train, validation, and test rows."""
    rng = np.random.default_rng(RANDOM_STATE)
    train, validation, test = [], [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) < 3:
            continue
        test_size = max(1, round(len(indices) * 0.1))
        validation_size = max(1, round(len(indices) * 0.1))
        test.extend(indices[:test_size])
        validation.extend(indices[test_size:test_size + validation_size])
        train.extend(indices[test_size + validation_size:])
    if not train or not validation or not test:
        raise ValueError("Each image category needs at least three examples.")
    return train, validation, test


def image_dataset(paths, labels, training):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            min(len(labels), 10_000),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True,
        )

    def load_image(path, label):
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        return tf.image.resize(image, IMAGE_SIZE), label

    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)


def fit_and_report(train, validation, test, class_names, input_shape, model_path, name, epochs):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = build_cnn(input_shape, len(class_names))
    history = model.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
        verbose=2,
    )
    loss, accuracy = model.evaluate(test, verbose=0)
    true = np.concatenate([labels.numpy() for _, labels in test])
    predicted = np.argmax(model.predict(test, verbose=0), axis=1)
    confusion = tf.math.confusion_matrix(
        true, predicted, num_classes=len(class_names)
    ).numpy()

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    (model_path.parent / f"{model_path.stem}_classes.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{
        "dataset": name,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "epochs_trained": len(history.epoch),
        "class_count": len(class_names),
    }]).to_csv(REPORT_DIR / f"day6_{name}_metrics.csv", index=False)
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
        REPORT_DIR / f"day6_{name}_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(max(7, len(class_names) * 0.45),) * 2)
    axis.imshow(confusion, cmap="Blues")
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted category",
        ylabel="True category",
        title=f"Day 6 — {name} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=90)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / f"day6_{name}_confusion_matrix.png", dpi=140)
    plt.close(figure)
    print(f"{name} test accuracy: {accuracy:.4f}")
    return model, {"test_loss": float(loss), "test_accuracy": float(accuracy)}, confusion


def train_fashion_mnist(model_path=FASHION_MODEL, epochs=20):
    (images, labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(labels))
    validation_size = len(indices) // 10
    validation, train = indices[:validation_size], indices[validation_size:]

    def make_dataset(x, y, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices(
            (x[..., np.newaxis], y)
        )
        if shuffle:
            dataset = dataset.shuffle(
                min(len(y), 10_000), seed=RANDOM_STATE
            )
        return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]
    return fit_and_report(
        make_dataset(images[train], labels[train], True),
        make_dataset(images[validation], labels[validation]),
        make_dataset(test_images, test_labels),
        classes,
        (28, 28, 1),
        model_path,
        "fashion_mnist",
        epochs,
    )


def train_catalog(data_dir, model_path=CATALOG_MODEL, epochs=20):
    data_dir = Path(data_dir)
    csv_path = data_dir / "styles.csv"
    if not csv_path.is_file():
        csv_path = data_dir / "myntradataset" / "styles.csv"
    images_dir = csv_path.parent / "images"
    if not csv_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected styles.csv and images/ under {data_dir}"
        )
    styles = pd.read_csv(csv_path, usecols=["id", "articleType"], dtype={"id": str})
    styles["path"] = styles["id"].map(lambda image_id: images_dir / f"{image_id}.jpg")
    styles = styles.dropna(subset=["articleType"])
    styles = styles[styles["path"].map(Path.is_file)]
    counts = styles["articleType"].value_counts()
    styles = styles[styles["articleType"].isin(counts[counts >= 3].index)]
    if styles.empty:
        raise ValueError("No usable product images found.")

    class_names = sorted(styles["articleType"].unique())
    class_ids = {name: index for index, name in enumerate(class_names)}
    labels = styles["articleType"].map(class_ids).to_numpy(dtype=np.int32)
    paths = styles["path"].map(str).to_numpy()
    train, validation, test = split_indices(labels)
    print(f"Using {len(labels)} images across {len(class_names)} categories.")
    return fit_and_report(
        image_dataset(paths[train], labels[train], True),
        image_dataset(paths[validation], labels[validation], False),
        image_dataset(paths[test], labels[test], False),
        class_names,
        (*IMAGE_SIZE, 3),
        model_path,
        "catalog",
        epochs,
    )


if __name__ == "__main__":
    # print("=== Fashion-MNIST CNN ===")
    # train_fashion_mnist()

    print("=== Product catalog CNN ===")
    train_catalog(DATA_DIR / "fashion-product-images-small")
