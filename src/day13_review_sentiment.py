"""Train a TF-IDF and Logistic Regression baseline for Amazon reviews.

Place a review CSV in data/amazon-reviews/train.csv, then run:
    python src/day13_review_sentiment.py
"""

import argparse
import csv
import itertools
from pathlib import Path

import joblib
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = (
    ROOT / "data" / "amazon-reviews" / "train.csv",
    ROOT / "data" / "amazon_polarity" / "train.csv",
    ROOT / "data" / "Reviews.csv",
)
MODEL_PATH = ROOT / "models" / "day13_review_sentiment.joblib"
METRICS_PATH = ROOT / "reports" / "day13_review_metrics.csv"
TERMS_PATH = ROOT / "reports" / "day13_informative_terms.csv"
LABELS = ("negative", "positive")
MAX_PER_CLASS = 3000
SEED = 42
TOKENIZER = RegexpTokenizer(r"[a-z]+")


def _english_stop_words():
    try:
        words = set(stopwords.words("english"))
    except LookupError as error:
        raise RuntimeError(
            "NLTK English stopwords are missing. Run "
            "`python -c \"import nltk; nltk.download('stopwords')\"`."
        ) from error
    return words - {"no", "not", "nor"}


def clean_text(text, stop_words=None):
    """Lowercase, tokenize, and remove English stopwords."""
    if stop_words is None:
        stop_words = _english_stop_words()
    tokens = TOKENIZER.tokenize(str(text).lower())
    return " ".join(token for token in tokens if token not in stop_words)


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

    column_name = column_name.lower()
    if column_name in {"score", "rating", "overall", "stars"}:
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


def load_reviews(csv_path, max_per_class=MAX_PER_CLASS):
    """Read a CSV and collect a balanced sample of positive and negative reviews."""
    if max_per_class < 1:
        raise ValueError("max_per_class must be at least 1.")

    csv_path = Path(csv_path)
    samples = {0: [], 1: []}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"No review rows found in {csv_path}.")

        normalized_columns = [name.strip().lower() for name in first_row]
        recognized_labels = {
            "label", "sentiment", "score", "rating", "overall", "stars"
        }
        recognized_text = {
            "title", "summary", "content", "review", "reviewtext", "text"
        }
        has_header = (
            bool(recognized_labels.intersection(normalized_columns))
            and bool(recognized_text.intersection(normalized_columns))
        )
        if has_header:
            label_index = next(
                index for index, name in enumerate(normalized_columns)
                if name in recognized_labels
            )
            text_indices = [
                index for index, name in enumerate(normalized_columns)
                if name in recognized_text
            ]
            label_column = normalized_columns[label_index]
            rows = reader
        else:
            if len(first_row) < 2:
                raise ValueError(
                    "A headerless review CSV must contain a label and review text."
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
            except ValueError:
                pass
            except IndexError:
                pass
        if 0 in numeric_labels:
            label_encoding = "zero_one"
        elif 1 in numeric_labels:
            label_encoding = "one_two"
        else:
            label_encoding = None

        for row in itertools.chain(first_rows, reader):
            label = _sentiment_label(
                row[label_index] if label_index < len(row) else "",
                label_column,
                label_encoding,
            )
            if label is None or len(samples[label]) >= max_per_class:
                if all(len(items) >= max_per_class for items in samples.values()):
                    break
                continue
            text = " ".join(
                str(row[index] or "").strip()
                for index in text_indices
                if index < len(row)
            )
            if text.strip():
                samples[label].append(text)

    counts = {LABELS[label]: len(texts) for label, texts in samples.items()}
    if min(map(len, samples.values())) < 3:
        raise ValueError(
            "Need at least three positive and three negative reviews for a "
            "stratified train/test split. "
            f"Loaded counts: {counts}"
        )
    print(f"Loaded balanced review sample: {counts}")
    texts = samples[0] + samples[1]
    labels = [0] * len(samples[0]) + [1] * len(samples[1])
    return texts, labels


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
        "Amazon review CSV not found. Place it in one of these locations or "
        f"pass --data:\n{expected}"
    )


def train_model(dataset_path=None, max_per_class=MAX_PER_CLASS,
                model_path=MODEL_PATH, report_dir=None):
    """Train, evaluate, and save the TF-IDF sentiment baseline."""
    dataset_path = _find_dataset(dataset_path)
    raw_texts, labels = load_reviews(dataset_path, max_per_class)
    stop_words = _english_stop_words()
    texts = [clean_text(text, stop_words) for text in raw_texts]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(x_train)
    test_features = vectorizer.transform(x_test)
    classifier = LogisticRegression(max_iter=500, random_state=SEED)
    classifier.fit(train_features, y_train)
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )

    report_dir = Path(report_dir) if report_dir else METRICS_PATH.parent
    model_path = Path(model_path)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": LABELS,
            "stop_words": list(stop_words),
        },
        model_path,
    )
    metrics_path = report_dir / "day13_review_metrics.csv"
    terms_path = report_dir / "day13_informative_terms.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "dataset", "train_reviews", "test_reviews", "accuracy",
            "negative_precision", "negative_recall",
            "positive_precision", "positive_recall",
        ])
        writer.writeheader()
        writer.writerow({
            "dataset": dataset_path.name,
            "train_reviews": len(y_train),
            "test_reviews": len(y_test),
            "accuracy": accuracy,
            "negative_precision": report["negative"]["precision"],
            "negative_recall": report["negative"]["recall"],
            "positive_precision": report["positive"]["precision"],
            "positive_recall": report["positive"]["recall"],
        })

    terms = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    negative_indices = sorted(range(len(terms)), key=coefficients.__getitem__)[:20]
    positive_indices = sorted(
        range(len(terms)), key=coefficients.__getitem__, reverse=True
    )[:20]
    with terms_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["sentiment", "term", "coefficient"]
        )
        writer.writeheader()
        for sentiment, indices in (
            ("positive", positive_indices),
            ("negative", negative_indices),
        ):
            for index in indices:
                writer.writerow({
                    "sentiment": sentiment,
                    "term": terms[index],
                    "coefficient": float(coefficients[index]),
                })

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved informative terms: {terms_path}")
    return accuracy


def predict_reviews(reviews, model_path=MODEL_PATH):
    """Predict sentiment and confidence for one review or a list of reviews."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Train the Day 13 model first: {model_path}")
    artifact = joblib.load(model_path)
    stop_words = set(artifact["stop_words"])
    single_review = isinstance(reviews, str)
    reviews = [reviews] if single_review else list(reviews)
    cleaned = [clean_text(review, stop_words) for review in reviews]
    features = artifact["vectorizer"].transform(cleaned)
    probabilities = artifact["classifier"].predict_proba(features)
    predictions = artifact["classifier"].classes_[probabilities.argmax(axis=1)]
    results = [
        {
            "sentiment": artifact["labels"][int(label)],
            "confidence": float(probability.max()),
        }
        for label, probability in zip(predictions, probabilities)
    ]
    return results[0] if single_review else results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to the Amazon review CSV")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=MAX_PER_CLASS,
        help=f"Maximum reviews per sentiment class (default: {MAX_PER_CLASS})",
    )
    args = parser.parse_args()
    if args.max_per_class < 3:
        parser.error("--max-per-class must be at least 3")
    train_model(args.data, args.max_per_class)


if __name__ == "__main__":
    main()
