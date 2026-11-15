"""Run Grad-CAM explanations for the Day 7 RetailMind product classifier.

Run from the repository root to explain three sample catalog photos:
    python src/day8_explainable_vision.py

Pass one or more image paths to choose specific photos:
    python src/day8_explainable_vision.py data/fashion-product-images-small/images/15970.jpg
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "data" / "fashion-product-images-small"
MODEL_PATHS = (
    (
        ROOT / "models" / "vision_master_category_resnet50.keras",
        ROOT / "models" / "vision_master_category_resnet50_classes.json",
    ),
    (
        ROOT / "models" / "vision_master_category_resnet50.keras",
        ROOT / "models" / "vision_master_category_class_names.json",
    ),
    (
        ROOT / "models" / "vision_master_category_resnet50_notebook.keras",
        ROOT / "models"
        / "vision_master_category_resnet50_notebook_classes.json",
    ),
    (
        ROOT / "models" / "vision_master_category_resnet50_notebook.keras",
        ROOT / "models" / "vision_master_category_class_names.json",
    ),
)
LAST_CONV_LAYER_NAME = "conv5_block3_out"
OUTPUT_DIR = ROOT / "reports" / "day8_gradcam"


@lru_cache(maxsize=1)
def _load_resources():
    for model_path, class_names_path in MODEL_PATHS:
        if model_path.is_file() and class_names_path.is_file():
            model = tf.keras.models.load_model(model_path)
            with class_names_path.open("r", encoding="utf-8") as file:
                class_names = json.load(file)
            if model.output_shape[-1] != len(class_names):
                raise ValueError(
                    "The saved model output does not match its class names."
                )
            return model, class_names

    expected_paths = "\n".join(str(path) for pair in MODEL_PATHS for path in pair)
    raise FileNotFoundError(
        "A Day 7 ResNet50 model and its class names were not found. "
        "Train the Day 7 transfer-learning model first. Checked:\n"
        f"{expected_paths}"
    )


def explain_image(image_path):
    """Return a Day 7 prediction and Grad-CAM heatmap for one product photo.

    The returned image is RGB in the 0-to-1 range and the heatmap is resized
    to the same height and width. Display them together with a Matplotlib
    alpha blend.
    """
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, class_names = _load_resources()
    input_shape = model.input_shape
    if len(input_shape) != 4 or input_shape[-1] != 3:
        raise ValueError("The Day 7 model must accept RGB images.")
    height, width = input_shape[1:3]
    if height is None or width is None:
        raise ValueError("The Day 7 model must use a fixed image size.")

    backbone = model.get_layer("resnet50")
    try:
        last_conv_layer = backbone.get_layer(LAST_CONV_LAYER_NAME)
    except ValueError as error:
        raise ValueError(
            f"ResNet50 layer '{LAST_CONV_LAYER_NAME}' was not found."
        ) from error

    feature_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output],
    )

    image = tf.keras.utils.load_img(
        image_path, target_size=(height, width), color_mode="rgb"
    )
    image = tf.keras.utils.img_to_array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    display_image = image / 255.0
    model_image = tf.keras.applications.resnet50.preprocess_input(
        tf.expand_dims(image, axis=0)
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = feature_model(
            [model_image], training=False
        )
        tape.watch(conv_outputs)
        predictions = backbone_features
        for layer in model.layers[2:]:
            predictions = layer(predictions, training=False)
        predicted_index = int(tf.argmax(predictions[0]).numpy())
        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise ValueError(
            "Could not calculate Grad-CAM gradients for the Day 7 model."
        )

    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(
        conv_outputs[0] * channel_weights, axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.math.divide_no_nan(
        heatmap, tf.reduce_max(heatmap)
    )
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (height, width)
    )[..., 0]

    return {
        "image": display_image,
        "heatmap": heatmap,
        "predicted_category": class_names[predicted_index],
        "confidence": float(predictions[0, predicted_index].numpy()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Show Grad-CAM explanations for Day 7 product predictions."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Product image paths. Defaults to the first three catalog JPGs.",
    )
    args = parser.parse_args()

    image_paths = [Path(path) for path in args.images]
    if not image_paths:
        image_dir = CATALOG_DIR / "images"
        if not image_dir.is_dir():
            image_dir = CATALOG_DIR / "myntradataset" / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(
                "Product photo folder not found. Expected "
                f"{CATALOG_DIR / 'images'} or "
                f"{CATALOG_DIR / 'myntradataset' / 'images'}."
            )
        image_paths = sorted(image_dir.glob("*.jpg"))[:3]
        if not image_paths:
            raise FileNotFoundError(f"No JPG product photos found in {image_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        result = explain_image(image_path)
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(result["image"])
        axes[0].set_title("Product photo")
        axes[1].imshow(result["image"])
        axes[1].imshow(result["heatmap"], cmap="jet", alpha=0.4)
        axes[1].set_title("Grad-CAM overlay")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{result['predicted_category']} "
            f"({result['confidence']:.1%} confidence)"
        )
        figure.tight_layout()
        output_path = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization: {output_path}")
        plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
