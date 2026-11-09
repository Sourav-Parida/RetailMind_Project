"""Day 6 CNN training and product-image prediction.

Run from the repository root with ``python src/day6_cnn.py``. Uncomment the
catalog training call at the bottom to train on the product image dataset.
"""

import csv
import json
import random
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fashion-product-images-small"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
FASHION_MODEL_PATH = MODEL_DIR / "fashion_mnist_prototype.keras"
CATALOG_MODEL_PATH = MODEL_DIR / "vision_catalog_classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "vision_class_names.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "vision_confusion_matrix.csv"

FASHION_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def build_fashion_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    model = build_fashion_mnist_model()
    model.summary()
    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Fashion-MNIST test loss: {test_loss:.4f}")
    print(f"Fashion-MNIST test accuracy: {test_accuracy:.4f}")
    model.save(FASHION_MODEL_PATH)
    print(f"Saved prototype model: {FASHION_MODEL_PATH}")

    predictions = tf.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = tf.math.confusion_matrix(y_test, predictions, num_classes=10)
    print("Fashion-MNIST confusion matrix (rows=true, columns=predicted):")
    print(matrix.numpy())
    print("Class order:", FASHION_CLASS_NAMES)


def load_catalog_records():
    """Read image paths and articleType labels from the catalog CSV."""
    csv_path = DATA_DIR / "styles.csv"
    if not csv_path.is_file():
        csv_path = DATA_DIR / "myntradataset" / "styles.csv"
    image_dir = csv_path.parent / "images"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find styles.csv under {DATA_DIR}. Download and "
            "extract the Fashion Product Images Small dataset there."
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Could not find image directory: {image_dir}")

    records = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "articleType"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "styles.csv must contain columns: id and articleType. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            image_id = (row.get("id") or "").strip()
            label = (row.get("articleType") or "").strip()
            if not image_id or not label:
                continue
            image_path = image_dir / f"{image_id}.jpg"
            if image_path.is_file():
                records.append((str(image_path), label))

    if not records:
        raise ValueError(
            "No matching image files found. Check styles.csv and images/*.jpg."
        )
    return records


def split_catalog_records(records):
    """Split each category into training, validation, and test samples."""
    grouped = {}
    for path, label in records:
        grouped.setdefault(label, []).append(path)

    kept = {label: paths for label, paths in grouped.items() if len(paths) >= 3}
    dropped = len(grouped) - len(kept)
    if dropped:
        print(f"Skipping {dropped} categories with fewer than 3 images.")

    class_names = sorted(kept)
    label_ids = {name: index for index, name in enumerate(class_names)}
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    rng = random.Random(SEED)

    for label in class_names:
        paths = kept[label][:]
        rng.shuffle(paths)
        count = len(paths)
        test_size = max(1, round(count * 0.15))
        val_size = max(1, round(count * 0.15))
        while test_size + val_size >= count:
            if test_size >= val_size and test_size > 1:
                test_size -= 1
            elif val_size > 1:
                val_size -= 1
            else:
                break

        test_part = paths[:test_size]
        val_part = paths[test_size:test_size + val_size]
        train_part = paths[test_size + val_size:]
        label_id = label_ids[label]
        train_paths.extend(train_part)
        train_labels.extend([label_id] * len(train_part))
        val_paths.extend(val_part)
        val_labels.extend([label_id] * len(val_part))
        test_paths.extend(test_part)
        test_labels.extend([label_id] * len(test_part))

    if not train_paths or not val_paths or not test_paths:
        raise ValueError("Not enough catalog images for train/validation/test splits.")
    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
        class_names,
    )


def decode_and_resize(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label


def make_catalog_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    return (
        dataset.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


def build_catalog_model(number_of_classes):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ], name="image_augmentation")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        augmentation,
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.40),
        tf.keras.layers.Dense(number_of_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_catalog():
    records = load_catalog_records()
    print(f"Usable images: {len(records)}")
    (train_data, val_data, test_data, class_names) = split_catalog_records(records)
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    print(f"Classes: {len(class_names)}")
    print(
        f"Train: {len(train_paths)} | "
        f"Validation: {len(val_paths)} | Test: {len(test_paths)}"
    )

    train_ds = make_catalog_dataset(train_paths, train_labels, training=True)
    val_ds = make_catalog_dataset(val_paths, val_labels)
    test_ds = make_catalog_dataset(test_paths, test_labels)
    model = build_catalog_model(len(class_names))
    model.summary()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=CATALOG_MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
    )

    model = tf.keras.models.load_model(CATALOG_MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Catalog test loss: {test_loss:.4f}")
    print(f"Catalog test accuracy: {test_accuracy:.4f}")

    y_true = tf.constant(test_labels, dtype=tf.int32)
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = tf.argmax(probabilities, axis=1, output_type=tf.int32)
    matrix = tf.math.confusion_matrix(
        y_true, y_pred, num_classes=len(class_names)
    ).numpy()
    with CONFUSION_MATRIX_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())
    with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print(f"Saved model: {CATALOG_MODEL_PATH}")
    print(f"Saved class labels: {CLASS_NAMES_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


def predict_catalog_image(image_path):
    """Print the predicted catalog category and confidence for an image."""
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not CATALOG_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Train the catalog classifier first. Expected model and class "
            "labels in the models/ directory."
        )

    model = tf.keras.models.load_model(CATALOG_MODEL_PATH)
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    probabilities = model.predict(tf.expand_dims(array, axis=0), verbose=0)[0]
    best_index = int(tf.argmax(probabilities))
    print(f"Predicted category: {class_names[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.2%}")


if __name__ == "__main__":
    # print("=== Fashion-MNIST CNN ===")
    # train_fashion_mnist()

    print("=== Product catalog CNN ===")
    train_catalog()

    # After training the catalog model, pass an image path to this function:
    # predict_catalog_image(DATA_DIR / "images" / "15970.jpg")
