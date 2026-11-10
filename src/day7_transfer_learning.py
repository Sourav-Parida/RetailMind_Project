"""Day 7 ResNet50 transfer learning for product catalog classification.

Train the Day 6 catalog CNN first, then run this file from the repository root:
    python src/day7_transfer_learning.py

YOLO/object detection is intentionally not included.
"""

import csv
import json
from pathlib import Path

import tensorflow as tf

if __package__:
    from .day6_cnn import (
        AUTOTUNE,
        BATCH_SIZE,
        CATALOG_MODEL_PATH,
        CLASS_NAMES_PATH,
        IMAGE_SIZE,
        SEED,
        decode_and_resize,
        load_catalog_records,
        make_catalog_dataset,
        split_catalog_records,
    )
else:
    from day6_cnn import (
        AUTOTUNE,
        BATCH_SIZE,
        CATALOG_MODEL_PATH,
        CLASS_NAMES_PATH,
        IMAGE_SIZE,
        SEED,
        decode_and_resize,
        load_catalog_records,
        make_catalog_dataset,
        split_catalog_records,
    )


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 3
TRANSFER_MODEL_PATH = MODEL_DIR / "vision_master_category_resnet50.keras"
TRANSFER_CLASS_NAMES_PATH = (
    MODEL_DIR / "vision_master_category_resnet50_classes.json"
)
COMPARISON_PATH = ROOT / "reports" / "day7_transfer_learning_comparison.csv"
CONFUSION_MATRIX_PATH = (
    MODEL_DIR / "day7_master_category_confusion_matrix.csv"
)


def make_transfer_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def preprocess(path, label):
        image, label = decode_and_resize(path, label)
        if training:
            image = tf.image.random_flip_left_right(image)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        return image, label

    return (
        dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


def build_transfer_model(number_of_classes, weights="imagenet"):
    """Build ResNet50 with a new classification head."""
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=(*IMAGE_SIZE, 3),
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(
        features
    )
    model = tf.keras.Model(inputs, outputs, name="retailmind_resnet50")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def train_day7_transfer_learning():
    """Train and compare a ResNet50 product classifier against Day 6."""
    tf.keras.utils.set_random_seed(SEED)
    if not CATALOG_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Train the Day 6 catalog CNN first. Expected "
            f"{CATALOG_MODEL_PATH.name} and {CLASS_NAMES_PATH.name} in models/."
        )

    records = load_catalog_records()
    (train_data, val_data, test_data, class_names) = split_catalog_records(
        records
    )
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        day6_class_names = json.load(file)
    if class_names != day6_class_names:
        raise ValueError(
            "The Day 6 class labels do not match the current catalog split. "
            "Retrain the Day 6 catalog model before comparing."
        )

    train_ds = make_transfer_dataset(train_paths, train_labels, training=True)
    val_ds = make_transfer_dataset(val_paths, val_labels)
    test_ds = make_transfer_dataset(test_paths, test_labels)

    day6_model = tf.keras.models.load_model(CATALOG_MODEL_PATH)
    if day6_model.input_shape[1:3] != IMAGE_SIZE:
        raise ValueError(
            "The Day 6 model input size does not match the Day 7 image size. "
            "Retrain the Day 6 catalog model before comparing."
        )
    if day6_model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The Day 6 model output does not match the catalog labels."
        )
    day6_loss, day6_accuracy = day6_model.evaluate(
        make_catalog_dataset(test_paths, test_labels), verbose=1
    )
    print(f"Day 6 CNN test accuracy: {day6_accuracy:.4f}")
    del day6_model
    tf.keras.backend.clear_session()

    model, backbone = build_transfer_model(len(class_names))
    model.summary()
    print("Training the new classifier head with the ResNet50 base frozen.")
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=TRANSFER_MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
    )
    best_head_val_accuracy = max(head_history.history["val_accuracy"])

    print("Fine-tuning the top ResNet50 convolutional block.")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = (
            layer.name.startswith("conv5_")
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=TRANSFER_MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                initial_value_threshold=best_head_val_accuracy,
            ),
        ],
    )

    best_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    transfer_loss, transfer_accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"Day 7 ResNet50 test loss: {transfer_loss:.4f}")
    print(f"Day 7 ResNet50 test accuracy: {transfer_accuracy:.4f}")

    predictions = best_model.predict(test_ds, verbose=1)
    matrix = tf.math.confusion_matrix(
        test_labels,
        tf.argmax(predictions, axis=1, output_type=tf.int32),
        num_classes=len(class_names),
    ).numpy()
    with CONFUSION_MATRIX_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())

    with TRANSFER_CLASS_NAMES_PATH.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["model", "test_loss", "test_accuracy"]
        )
        writer.writeheader()
        writer.writerow({
            "model": "Day 6 catalog CNN (masterCategory)",
            "test_loss": f"{day6_loss:.6f}",
            "test_accuracy": f"{day6_accuracy:.6f}",
        })
        writer.writerow({
            "model": "Day 7 ResNet50 transfer learning (masterCategory)",
            "test_loss": f"{transfer_loss:.6f}",
            "test_accuracy": f"{transfer_accuracy:.6f}",
        })
    print(f"Saved Day 7 model: {TRANSFER_MODEL_PATH}")
    print(f"Saved Day 7 labels: {TRANSFER_CLASS_NAMES_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


def predict_catalog_image(image_path):
    """Print the Day 7 category and confidence for one product image."""
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not TRANSFER_MODEL_PATH.is_file() or not TRANSFER_CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            "Train the Day 7 transfer-learning classifier first."
        )

    model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    with TRANSFER_CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    array = tf.keras.applications.resnet50.preprocess_input(array)
    probabilities = model.predict(tf.expand_dims(array, axis=0), verbose=0)[0]
    best_index = int(tf.argmax(probabilities))
    print(f"Predicted master category: {class_names[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.2%}")


if __name__ == "__main__":
    train_day7_transfer_learning()
