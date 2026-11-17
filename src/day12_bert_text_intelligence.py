"""Fine-tune a BERT-family model for review sentiment and ticket routing.

Run from the repository root:
    python src/day12_bert_text_intelligence.py
"""

import argparse
import csv
import itertools
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

os.environ.setdefault("USE_TF", "0")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = (
    ROOT / "data" / "amazon-reviews" / "train.csv",
    ROOT / "data" / "amazon_polarity" / "train.csv",
    ROOT / "data" / "Reviews.csv",
)
BASELINE_METRICS_PATH = ROOT / "reports" / "day13_review_metrics.csv"
SENTIMENT_MODEL_DIR = ROOT / "models" / "day12_bert_sentiment"
ROUTER_MODEL_DIR = ROOT / "models" / "day12_bert_ticket_router"
METRICS_PATH = ROOT / "reports" / "day12_bert_metrics.csv"
COMPARISON_PATH = ROOT / "reports" / "day12_bert_comparison.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_PER_CLASS = 3000
TICKET_MAX_PER_CLASS = 128
MAX_LENGTH = 128
EPOCHS = 1
SEED = 42
SENTIMENT_LABELS = ("negative", "positive")
TICKET_LABELS = (
    "shipping_delivery",
    "returns_refunds",
    "product_quality",
    "general_product_feedback",
)
LABEL_COLUMNS = {"label", "sentiment", "score", "rating", "overall", "stars"}
TEXT_COLUMNS = {"title", "summary", "content", "review", "reviewtext", "text"}
RATING_COLUMNS = {"score", "rating", "overall", "stars"}
ROUTING_KEYWORDS = {
    "shipping_delivery": (
        "ship", "deliver", "arriv", "package", "shipment", "tracking", "late",
    ),
    "returns_refunds": (
        "refund", "return", "exchange", "money back", "replacement", "replace",
    ),
    "product_quality": (
        "broken", "defect", "quality", "damaged", "faulty", "broke",
        "stopped working", "fell apart", "poorly made",
    ),
}


def _find_dataset(dataset_path=None):
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Review dataset CSV not found: {path}")
        return path
    for path in DATA_PATHS:
        if path.is_file():
            return path
    expected = "\n".join(str(path) for path in DATA_PATHS)
    raise FileNotFoundError(
        "Amazon review CSV not found. Place it in data/amazon-reviews/ or "
        f"pass --data:\n{expected}"
    )


def _sentiment_label(value, column_name, label_encoding):
    value = str(value).strip().lower()
    if value in {"negative", "neg", "bad"}:
        return 0
    if value in {"positive", "pos", "good"}:
        return 1
    try:
        score = float(value)
    except ValueError:
        return None

    if column_name in RATING_COLUMNS:
        if score <= 2:
            return 0
        if score >= 4:
            return 1
        return None
    if label_encoding == "zero_one":
        return int(score) if score in {0, 1} else None
    if label_encoding == "one_two":
        return int(score == 2) if score in {1, 2} else None
    return None


def load_review_examples(csv_path, max_per_class=MAX_PER_CLASS):
    """Load a balanced raw-text sample from headed or headerless CSV files."""
    if max_per_class < 1:
        raise ValueError("max_per_class must be at least 1.")

    samples = {0: [], 1: []}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"No review rows found in {csv_path}.")

        normalized_columns = [column.strip().lower() for column in first_row]
        has_header = (
            bool(LABEL_COLUMNS.intersection(normalized_columns))
            and bool(TEXT_COLUMNS.intersection(normalized_columns))
        )
        if has_header:
            label_index = next(
                index for index, name in enumerate(normalized_columns)
                if name in LABEL_COLUMNS
            )
            text_indices = [
                index for index, name in enumerate(normalized_columns)
                if name in TEXT_COLUMNS
            ]
            label_column = normalized_columns[label_index]
            rows = reader
        else:
            if len(first_row) < 2:
                raise ValueError(
                    "A headerless review CSV must contain a label and text."
                )
            label_index = 0
            text_indices = list(range(1, len(first_row)))
            label_column = "label"
            rows = itertools.chain((first_row,), reader)

        first_rows = list(itertools.islice(rows, 1000))
        numeric_labels = set()
        for row in first_rows:
            try:
                numeric_labels.add(float(row[label_index]))
            except (IndexError, ValueError):
                continue
        if 0 in numeric_labels:
            label_encoding = "zero_one"
        elif 1 in numeric_labels or 2 in numeric_labels:
            label_encoding = "one_two"
        else:
            label_encoding = None

        for row in itertools.chain(first_rows, rows):
            raw_label = row[label_index] if label_index < len(row) else ""
            label = _sentiment_label(
                raw_label, label_column, label_encoding
            )
            if label is None or len(samples[label]) >= max_per_class:
                if all(len(items) >= max_per_class for items in samples.values()):
                    break
                continue
            text = " ".join(
                row[index].strip()
                for index in text_indices
                if index < len(row) and row[index].strip()
            )
            if text:
                samples[label].append(text)

    counts = {
        SENTIMENT_LABELS[label]: len(texts)
        for label, texts in samples.items()
    }
    if min(map(len, samples.values())) < 3:
        raise ValueError(
            "Need at least three examples for each sentiment. "
            f"Loaded counts: {counts}"
        )
    print(f"Loaded review sample: {counts}")
    texts = samples[0] + samples[1]
    labels = [0] * len(samples[0]) + [1] * len(samples[1])
    return texts, labels


def build_ticket_examples(review_texts, max_per_class=TICKET_MAX_PER_CLASS):
    """Create weakly labeled routing examples from review text.

    These rules create a demonstration dataset, not ground-truth ticket labels.
    """
    samples = {label: [] for label in TICKET_LABELS}
    for text in review_texts:
        normalized_text = text.lower()
        category = next(
            (
                label for label, keywords in ROUTING_KEYWORDS.items()
                if any(keyword in normalized_text for keyword in keywords)
            ),
            "general_product_feedback",
        )
        if len(samples[category]) < max_per_class:
            samples[category].append(text)

    counts = {label: len(texts) for label, texts in samples.items()}
    if min(counts.values()) < 3:
        raise ValueError(
            "Not enough review-derived examples for every ticket category. "
            f"Loaded counts: {counts}. Increase --max-per-class or use a "
            "larger review CSV."
        )
    print(f"Weakly labeled ticket sample: {counts}")
    texts = [text for label in TICKET_LABELS for text in samples[label]]
    labels = [
        index
        for index, label in enumerate(TICKET_LABELS)
        for _ in samples[label]
    ]
    return texts, labels


def _stratified_split(texts, labels, seed=SEED, test_fraction=0.2):
    """Make a deterministic per-class split without another ML dependency."""
    grouped = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices, test_indices = [], []
    for label_indices in grouped.values():
        if len(label_indices) < 2:
            raise ValueError("Every class needs at least two examples.")
        rng.shuffle(label_indices)
        test_count = max(1, math.ceil(len(label_indices) * test_fraction))
        test_count = min(test_count, len(label_indices) - 1)
        test_indices.extend(label_indices[:test_count])
        train_indices.extend(label_indices[test_count:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        [texts[index] for index in train_indices],
        [texts[index] for index in test_indices],
        [labels[index] for index in train_indices],
        [labels[index] for index in test_indices],
    )


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = self.labels[index]
        return encoded


def _class_metrics(true_labels, predicted_labels, label_names):
    total = len(true_labels)
    correct = sum(
        truth == prediction
        for truth, prediction in zip(true_labels, predicted_labels)
    )
    rows = []
    f1_scores = []
    for label_id, label_name in enumerate(label_names):
        true_positive = sum(
            truth == label_id and prediction == label_id
            for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_positive = sum(
            truth != label_id and prediction == label_id
            for truth, prediction in zip(true_labels, predicted_labels)
        )
        false_negative = sum(
            truth == label_id and prediction != label_id
            for truth, prediction in zip(true_labels, predicted_labels)
        )
        support = sum(truth == label_id for truth in true_labels)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        f1_scores.append(f1)
        rows.append({
            "label": label_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_class": rows,
    }


def _train_classifier(
    texts,
    labels,
    label_names,
    model_dir,
    model_name,
    max_length,
    epochs,
    seed,
):
    train_texts, test_texts, train_labels, test_labels = _stratified_split(
        texts, labels, seed
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: index for index, name in enumerate(label_names)},
    )
    train_dataset = ReviewDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    test_dataset = ReviewDataset(
        test_texts, test_labels, tokenizer, max_length
    )
    training_args = TrainingArguments(
        output_dir=str(model_dir / "training"),
        num_train_epochs=epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        dataloader_pin_memory=False,
        seed=seed,
        data_seed=seed,
        fp16=torch.cuda.is_available(),
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    predictions = trainer.predict(test_dataset).predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_labels = [
        max(range(len(scores)), key=scores.__getitem__)
        for scores in predictions.tolist()
    ]
    metrics = _class_metrics(test_labels, predicted_labels, label_names)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.config.max_length = max_length
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    return metrics, len(train_labels), len(test_labels)


def _read_classical_baseline():
    if not BASELINE_METRICS_PATH.is_file():
        raise FileNotFoundError(
            "Run the existing classical baseline first: "
            "python src/day13_review_sentiment.py. "
            f"Expected its metrics at {BASELINE_METRICS_PATH}."
        )
    with BASELINE_METRICS_PATH.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file), None)
    if not row or not row.get("accuracy"):
        raise ValueError(
            f"Classical baseline metrics are invalid: {BASELINE_METRICS_PATH}"
        )
    return float(row["accuracy"])


def _write_metrics(path, task, metrics, train_count, test_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task", "label", "accuracy", "macro_f1", "precision",
                "recall", "f1", "train_examples", "test_examples",
            ],
        )
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow({
                "task": task,
                "label": row["label"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "train_examples": train_count,
                "test_examples": test_count,
            })


def train_text_intelligence(
    dataset_path=None,
    max_per_class=MAX_PER_CLASS,
    ticket_max_per_class=TICKET_MAX_PER_CLASS,
    model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
    epochs=EPOCHS,
):
    """Train sentiment and review-derived support-ticket classifiers."""
    baseline_accuracy = _read_classical_baseline()
    dataset_path = _find_dataset(dataset_path)
    review_texts, sentiment_labels = load_review_examples(
        dataset_path, max_per_class
    )
    sentiment_metrics, sentiment_train_count, sentiment_test_count = (
        _train_classifier(
            review_texts,
            sentiment_labels,
            SENTIMENT_LABELS,
            SENTIMENT_MODEL_DIR,
            model_name,
            max_length,
            epochs,
            SEED,
        )
    )
    ticket_texts, ticket_labels = build_ticket_examples(
        review_texts, ticket_max_per_class
    )
    ticket_metrics, ticket_train_count, ticket_test_count = _train_classifier(
        ticket_texts,
        ticket_labels,
        TICKET_LABELS,
        ROUTER_MODEL_DIR,
        model_name,
        max_length,
        epochs,
        SEED,
    )

    _write_metrics(
        METRICS_PATH,
        "sentiment",
        sentiment_metrics,
        sentiment_train_count,
        sentiment_test_count,
    )
    _write_metrics(
        ROOT / "reports" / "day12_ticket_routing_metrics.csv",
        "ticket_routing_weak_labels",
        ticket_metrics,
        ticket_train_count,
        ticket_test_count,
    )
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model", "accuracy", "macro_f1", "test_examples",
                "comparison_note",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "model": "Day 13 TF-IDF Logistic Regression baseline",
            "accuracy": baseline_accuracy,
            "macro_f1": "",
            "test_examples": "",
            "comparison_note": "Day 13 saved holdout; separate from Day 12 split",
        })
        writer.writerow({
            "model": f"{model_name} fine-tuned sentiment",
            "accuracy": sentiment_metrics["accuracy"],
            "macro_f1": sentiment_metrics["macro_f1"],
            "test_examples": sentiment_test_count,
            "comparison_note": "Day 12 held-out split; not the same review rows",
        })

    print(
        f"Classical TF-IDF baseline accuracy: {baseline_accuracy:.4f} "
        "(reference holdout)"
    )
    print(
        f"BERT sentiment accuracy: {sentiment_metrics['accuracy']:.4f}; "
        f"macro-F1: {sentiment_metrics['macro_f1']:.4f}"
    )
    print(
        "Ticket routing accuracy: "
        f"{ticket_metrics['accuracy']:.4f}; "
        f"macro-F1: {ticket_metrics['macro_f1']:.4f}"
    )
    print(f"Saved BERT comparison: {COMPARISON_PATH}")
    return {
        "sentiment": sentiment_metrics,
        "ticket_routing": ticket_metrics,
        "baseline_accuracy": baseline_accuracy,
    }


def _predict(texts, model_dir, single_text):
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Train the Day 12 model first: {model_dir}"
        )
    texts = [texts] if single_text else list(texts)
    if not texts:
        return []
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    max_length = getattr(model.config, "max_length", MAX_LENGTH)
    results = []
    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
        for row in probabilities.cpu().tolist():
            index = max(range(len(row)), key=row.__getitem__)
            label = model.config.id2label[index]
            results.append({
                "label": label,
                "confidence": float(row[index]),
            })
    return results[0] if single_text else results


def predict_sentiment(texts):
    """Predict sentiment for one review or a sequence of reviews."""
    return _predict(texts, SENTIMENT_MODEL_DIR, isinstance(texts, str))


def route_support_ticket(texts):
    """Predict a routing category for one ticket or a sequence of tickets."""
    return _predict(texts, ROUTER_MODEL_DIR, isinstance(texts, str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to Amazon review CSV")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=MAX_PER_CLASS,
        help=f"Maximum reviews per sentiment class (default: {MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--ticket-max-per-class",
        type=int,
        default=TICKET_MAX_PER_CLASS,
        help=(
            "Maximum review-derived tickets per category "
            f"(default: {TICKET_MAX_PER_CLASS})"
        ),
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if min(args.max_per_class, args.ticket_max_per_class) < 3:
        parser.error("Sample limits must be at least 3.")
    if args.max_length < 8 or args.epochs < 1:
        parser.error("--max-length must be at least 8 and --epochs at least 1.")
    train_text_intelligence(
        dataset_path=args.data,
        max_per_class=args.max_per_class,
        ticket_max_per_class=args.ticket_max_per_class,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
