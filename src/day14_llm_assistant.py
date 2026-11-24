"""Day 14: review theme summaries and a grounded retail support assistant.

Examples (set OPENAI_API_KEY in the environment before chat):
    python -m src.day14_llm_assistant summarize --data data/amazon-reviews/train.csv
    python -m src.day14_llm_assistant chat --knowledge data/product_policy.txt

Review CSVs should include a review text column. Product-specific summaries
also require a product identifier/name column; Amazon Polarity itself has no
product identifiers, so its reviews are grouped as one dataset when absent.
"""

import argparse
import csv
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = (
    ROOT / "data" / "amazon-reviews" / "train.csv",
    ROOT / "data" / "amazon_polarity" / "train.csv",
    ROOT / "data" / "Reviews.csv",
)
SUMMARY_PATH = ROOT / "reports" / "day14_review_summaries.json"
REVIEW_COLUMNS = ("review", "reviewtext", "text", "content", "body", "summary")
PRODUCT_COLUMNS = ("product_id", "productid", "asin", "product", "product_name", "item")


def _find_reviews(path=None):
    if path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Review CSV not found: {candidate}")
        return candidate
    for candidate in DATA_PATHS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Review CSV not found; pass --data or place it under data/amazon-reviews/.")


def load_product_reviews(path):
    """Load review text grouped by product, accepting common CSV column names."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Review CSV must have a header row.")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        review_column = next((columns[name] for name in REVIEW_COLUMNS if name in columns), None)
        product_column = next((columns[name] for name in PRODUCT_COLUMNS if name in columns), None)
        title_column = next((columns[name] for name in ("title", "review_title") if name in columns), None)
        if review_column is None:
            raise ValueError(f"Review text column missing. Expected one of: {', '.join(REVIEW_COLUMNS)}")
        grouped = defaultdict(list)
        for row in reader:
            review = " ".join(part.strip() for part in (
                row.get(title_column, "") if title_column else "", row.get(review_column, "")
            ) if part and part.strip())
            if review:
                product = (row.get(product_column, "").strip() if product_column else "") or "all_reviews"
                grouped[product].append(review)
    if not grouped:
        raise ValueError("No non-empty reviews found in the CSV.")
    return dict(grouped)


def summarize_reviews(reviews, model_name="sshleifer/distilbart-cnn-12-6", max_reviews=80):
    """Summarize reviews per product with a Hugging Face Transformers pipeline."""
    try:
        from transformers import pipeline
    except ImportError as error:
        raise RuntimeError("Install the project requirements to use Transformers summarization.") from error
    summarizer = pipeline("summarization", model=model_name)
    results = {}
    for product, items in reviews.items():
        text = "\n".join(items[:max_reviews])
        if len(text.split()) < 20:
            results[product] = "Not enough review text to identify reliable themes."
            continue
        # Keep each inference request within the model's input context.
        chunks, current = [], []
        for sentence in text.splitlines():
            if sum(map(len, current)) + len(sentence) > 2600 and current:
                chunks.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            chunks.append(" ".join(current))
        partials = []
        for chunk in chunks[:8]:
            output = summarizer(chunk, max_length=110, min_length=25, do_sample=False, truncation=True)
            partials.append(output[0]["summary_text"])
        combined = " ".join(partials)
        if len(partials) > 1:
            combined = summarizer(combined, max_length=120, min_length=30, do_sample=False, truncation=True)[0]["summary_text"]
        results[product] = combined
    return results


class OpenAIChatClient:
    """Small OpenAI-compatible chat-completions client using only stdlib."""

    def __init__(self, api_key=None, model=None, base_url=None, timeout=45):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY in your environment before starting chat.")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout

    def complete(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.2}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM API returned HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach the LLM API: {error.reason}") from error
        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("LLM API response did not contain a chat completion.") from error


class RetailSupportAssistant:
    """History-aware customer support grounded in supplied product/policy text."""

    def __init__(self, knowledge, client=None, history_limit=12):
        self.knowledge = str(knowledge).strip()
        if not self.knowledge:
            raise ValueError("Provide product or policy text to ground support answers.")
        self.client = client or OpenAIChatClient()
        self.history = []
        self.history_limit = max(2, history_limit)

    def respond(self, message):
        message = str(message).strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        system = (
            "You are RetailMind's friendly, concise customer support assistant. "
            "Answer only from the supplied product and policy information. "
            "Do not invent prices, stock, delivery promises, or policy exceptions. "
            "If the information is missing, say so and suggest contacting support. "
            "Keep a helpful, respectful retail brand voice.\n\n"
            f"PRODUCT AND POLICY INFORMATION:\n{self.knowledge[:12000]}"
        )
        messages = [{"role": "system", "content": system}, *self.history, {"role": "user", "content": message}]
        answer = self.client.complete(messages)
        self.history.extend(({"role": "user", "content": message}, {"role": "assistant", "content": answer}))
        self.history = self.history[-self.history_limit:]
        return answer


def _save_summaries(summaries, path=SUMMARY_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize", help="Summarize review themes by product")
    summarize.add_argument("--data", help="Review CSV path")
    summarize.add_argument("--model", default="sshleifer/distilbart-cnn-12-6", help="Hugging Face summarization model")
    summarize.add_argument("--output", default=str(SUMMARY_PATH), help="Output JSON path")
    chat = subparsers.add_parser("chat", help="Start the grounded support chatbot")
    chat.add_argument("--knowledge", required=True, help="Text file with product and policy details")
    chat.add_argument("--model", help="OpenAI-compatible model (default: OPENAI_MODEL or gpt-4o-mini)")
    chat.add_argument("--base-url", help="Optional OpenAI-compatible API base URL")
    args = parser.parse_args()
    if args.command == "summarize":
        groups = load_product_reviews(_find_reviews(args.data))
        summaries = summarize_reviews(groups, model_name=args.model)
        output = _save_summaries(summaries, args.output)
        for product, summary in summaries.items():
            print(f"\n{product}: {summary}")
        print(f"\nSaved summaries: {output}")
    else:
        knowledge_path = Path(args.knowledge)
        if not knowledge_path.is_absolute():
            knowledge_path = ROOT / knowledge_path
        if not knowledge_path.is_file():
            parser.error(f"Knowledge file not found: {knowledge_path}")
        client = OpenAIChatClient(model=args.model, base_url=args.base_url)
        assistant = RetailSupportAssistant(knowledge_path.read_text(encoding="utf-8"), client)
        print("RetailMind Support (type 'quit' to exit)")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"quit", "exit"}:
                break
            if question:
                try:
                    print(f"RetailMind: {assistant.respond(question)}")
                except RuntimeError as error:
                    print(f"Request failed: {error}")


if __name__ == "__main__":
    main()
