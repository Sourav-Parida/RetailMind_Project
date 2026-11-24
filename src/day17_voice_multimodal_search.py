"""Day 17: spoken support replies and CLIP image/text catalog search.

Build the catalog index once:
    python -m src.day17_voice_multimodal_search build-index

Search the index:
    python -m src.day17_voice_multimodal_search search-text --query "blue shirt for summer"
    python -m src.day17_voice_multimodal_search search-image --image path/to/query.jpg

Create speech from a reply:
    python -m src.day17_voice_multimodal_search speak --text "Your order is on its way."

The first run downloads the CLIP/TTS checkpoints from Hugging Face. The
precomputed catalog embeddings are local and are not committed to Git.
"""

import argparse
import csv
import json
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "fashion-product-images-small"
DEFAULT_INDEX_DIR = ROOT / "models" / "day17_clip_catalog"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TTS_MODEL = "facebook/mms-tts-eng"


def _resolve_catalog_dir(path=DEFAULT_DATA_DIR):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if (path / "styles.csv").is_file():
        return path
    nested = path / "fashion-product-images-small"
    if (nested / "styles.csv").is_file():
        return nested
    raise FileNotFoundError(f"styles.csv not found in {path} or {nested}.")


def load_catalog(data_dir=DEFAULT_DATA_DIR, limit=None):
    """Read product metadata and retain rows with an existing catalog image."""
    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be at least 1 when specified.")
    data_dir = _resolve_catalog_dir(data_dir)
    image_dir = data_dir / "images"
    rows = []
    with (data_dir / "styles.csv").open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("styles.csv must contain an 'id' column.")
        for row in reader:
            product_id = str(row.get("id", "")).strip()
            if not product_id:
                continue
            image_path = image_dir / f"{product_id}.jpg"
            if not image_path.is_file():
                continue
            normalized = {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            normalized["id"] = product_id
            try:
                normalized["image_path"] = str(image_path.relative_to(ROOT))
            except ValueError:
                normalized["image_path"] = str(image_path)
            rows.append(normalized)
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise ValueError(f"No product rows with image files found under {data_dir}.")
    return rows


def _product_text(row):
    parts = [
        row.get("productDisplayName", ""),
        row.get("gender", ""), row.get("masterCategory", ""),
        row.get("subCategory", ""), row.get("articleType", ""),
        row.get("baseColour", ""), row.get("season", ""),
        row.get("usage", ""),
    ]
    return ", ".join(part for part in parts if part)


def _load_clip(model_name=DEFAULT_CLIP_MODEL, device=None):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise RuntimeError("Install the project requirements including PyTorch and Transformers to run CLIP.") from error
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, torch, device


def _normalize_features(features):
    features = features.float()
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def build_catalog_index(
    data_dir=DEFAULT_DATA_DIR,
    output_dir=DEFAULT_INDEX_DIR,
    model_name=DEFAULT_CLIP_MODEL,
    batch_size=24,
    limit=None,
    device=None,
):
    """Embed product photos and descriptive metadata into a shared CLIP space."""
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1.")
    rows = load_catalog(data_dir, limit)
    model, processor, torch, device = _load_clip(model_name, device)
    image_features, text_features, valid_rows = [], [], []
    started = time.perf_counter()
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start:start + int(batch_size)]
        loaded_images, kept = [], []
        for row in batch:
            try:
                with Image.open(row["image_path"]) as image:
                    loaded_images.append(image.convert("RGB"))
                    kept.append(row)
            except (OSError, ValueError):
                continue
        if not kept:
            continue
        with torch.inference_mode():
            image_inputs = processor(images=loaded_images, return_tensors="pt").to(device)
            image_vectors = _normalize_features(model.get_image_features(**image_inputs))
            descriptions = [_product_text(row) for row in kept]
            text_inputs = processor(text=descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
            text_vectors = _normalize_features(model.get_text_features(**text_inputs))
        image_features.append(image_vectors.cpu().numpy().astype(np.float16))
        text_features.append(text_vectors.cpu().numpy().astype(np.float16))
        valid_rows.extend(kept)
    if not valid_rows:
        raise ValueError("No readable product photos were found in the selected catalog rows.")

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "clip_catalog_embeddings.npz"
    np.savez(
        embedding_path,
        image_embeddings=np.concatenate(image_features),
        text_embeddings=np.concatenate(text_features),
    )
    products_path = output_dir / "clip_catalog_products.json"
    products_path.write_text(json.dumps(valid_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "model_name": model_name,
        "device": device,
        "indexed_products": len(valid_rows),
        "catalog_rows_without_image_or_readable_photo": len(rows) - len(valid_rows),
        "embedding_dimensions": int(image_features[0].shape[-1]),
        "embedding_dtype_on_disk": "float16",
        "index_build_seconds": time.perf_counter() - started,
        "embedding_file": str(embedding_path),
        "product_metadata_file": str(products_path),
    }
    report_path = output_dir / "clip_catalog_index_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _load_index(index_dir=DEFAULT_INDEX_DIR):
    index_dir = Path(index_dir)
    if not index_dir.is_absolute():
        index_dir = ROOT / index_dir
    embedding_path = index_dir / "clip_catalog_embeddings.npz"
    products_path = index_dir / "clip_catalog_products.json"
    if not embedding_path.is_file() or not products_path.is_file():
        raise FileNotFoundError("CLIP index is missing. Run the build-index command first.")
    embeddings = np.load(embedding_path)
    return index_dir, embeddings["image_embeddings"], embeddings["text_embeddings"], json.loads(products_path.read_text(encoding="utf-8"))


def _top_results(scores, products, top_k):
    top_k = max(1, min(int(top_k), len(products)))
    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
    ranked = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    results = []
    for index in ranked:
        item = products[int(index)].copy()
        item["similarity"] = float(scores[int(index)])
        results.append(item)
    return results


def _embed_texts(queries, model, processor, torch, device):
    encoded = processor(text=queries, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        features = _normalize_features(model.get_text_features(**encoded))
    return features.cpu().numpy().astype(np.float32)


def _embed_images(images, model, processor, torch, device):
    encoded = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = _normalize_features(model.get_image_features(**encoded))
    return features.cpu().numpy().astype(np.float32)


def search_by_text(query, top_k=5, index_dir=DEFAULT_INDEX_DIR, model_name=None, image_weight=0.7, device=None):
    """Find catalog items using a natural-language product description."""
    if not str(query).strip():
        raise ValueError("query cannot be empty.")
    if not 0.0 <= float(image_weight) <= 1.0:
        raise ValueError("image_weight must be between 0 and 1.")
    index_dir, image_embeddings, text_embeddings, products = _load_index(index_dir)
    report = json.loads((index_dir / "clip_catalog_index_report.json").read_text(encoding="utf-8"))
    model, processor, torch, device = _load_clip(model_name or report["model_name"], device)
    query_vector = _embed_texts([str(query)], model, processor, torch, device)[0]
    image_scores = image_embeddings.astype(np.float32) @ query_vector
    text_scores = text_embeddings.astype(np.float32) @ query_vector
    scores = float(image_weight) * image_scores + (1.0 - float(image_weight)) * text_scores
    return _top_results(scores, products, top_k)


def search_by_image(image_path, top_k=5, index_dir=DEFAULT_INDEX_DIR, model_name=None, image_weight=0.7, device=None):
    """Find visually or semantically similar products from a query photo."""
    if not 0.0 <= float(image_weight) <= 1.0:
        raise ValueError("image_weight must be between 0 and 1.")
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Query image not found: {image_path}")
    index_dir, image_embeddings, text_embeddings, products = _load_index(index_dir)
    report = json.loads((index_dir / "clip_catalog_index_report.json").read_text(encoding="utf-8"))
    model, processor, torch, device = _load_clip(model_name or report["model_name"], device)
    with Image.open(image_path) as image:
        query_vector = _embed_images([image.convert("RGB")], model, processor, torch, device)[0]
    image_scores = image_embeddings.astype(np.float32) @ query_vector
    text_scores = text_embeddings.astype(np.float32) @ query_vector
    scores = float(image_weight) * image_scores + (1.0 - float(image_weight)) * text_scores
    return _top_results(scores, products, top_k)


class TextToSpeechSpeaker:
    """Reusable Hugging Face TTS pipeline with WAV output using Python stdlib."""

    def __init__(self, model_name=DEFAULT_TTS_MODEL, device=None):
        try:
            import torch
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError("Install the project requirements including PyTorch and Transformers to run TTS.") from error
        device = device or (0 if torch.cuda.is_available() else -1)
        self.model_name = model_name
        self.pipeline = pipeline("text-to-speech", model=model_name, device=device)

    def speak(self, text, output_path=None):
        """Synthesize one reply to a WAV file and return its audio metadata."""
        text = str(text).strip()
        if not text:
            raise ValueError("Speech text cannot be empty.")
        result = self.pipeline(text)
        audio = np.asarray(result["audio"], dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype("<i2")
        rate = int(result["sampling_rate"])
        if output_path is None:
            output_path = ROOT / "reports" / "day17_voice_reply.wav"
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(rate)
            audio_file.writeframes(pcm.tobytes())
        return {"output_path": str(output_path), "sampling_rate": rate, "duration_seconds": len(pcm) / rate, "model_name": self.model_name}


class VoiceSupportAssistant:
    """Attach spoken replies to the existing Day 14 support assistant."""

    def __init__(self, support_assistant, speaker=None):
        self.support_assistant = support_assistant
        self.speaker = speaker or TextToSpeechSpeaker()

    def respond(self, message, audio_path=None):
        answer = self.support_assistant.respond(message)
        audio = self.speaker.speak(answer, audio_path)
        return {"answer": answer, "audio": audio}


def _print_results(results):
    for rank, product in enumerate(results, start=1):
        print(f"{rank}. [{product['similarity']:.3f}] {product.get('productDisplayName') or product['id']}")
        print(f"   {product.get('masterCategory', '')} / {product.get('articleType', '')} / {product.get('baseColour', '')}")
        print(f"   {product['image_path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-index", help="Embed catalog images and text with CLIP")
    build.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    build.add_argument("--output-dir", default=str(DEFAULT_INDEX_DIR))
    build.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    build.add_argument("--batch-size", type=int, default=24)
    build.add_argument("--limit", type=int, help="Optional small product cap for a quick demo")
    for command, help_text in (("search-text", "Search catalog by text"), ("search-image", "Search catalog by image")):
        search = subparsers.add_parser(command, help=help_text)
        search.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
        search.add_argument("--top-k", type=int, default=5)
        search.add_argument("--image-weight", type=float, default=0.7)
        search.add_argument("--device", help="Optional torch device, e.g. cpu or cuda:0")
        if command == "search-text":
            search.add_argument("--query", required=True)
        else:
            search.add_argument("--image", required=True)
    speak = subparsers.add_parser("speak", help="Speak a supplied support reply")
    speak.add_argument("--text", required=True)
    speak.add_argument("--output", default="reports/day17_voice_reply.wav")
    speak.add_argument("--model", default=DEFAULT_TTS_MODEL)
    speak.add_argument("--device", help="Optional CUDA device index; CPU is selected when omitted")
    reply = subparsers.add_parser("speak-reply", help="Answer using the Day 14 support assistant, then speak the reply")
    reply.add_argument("--question", required=True)
    reply.add_argument("--knowledge", required=True, help="Product/policy text file")
    reply.add_argument("--audio-output", default="reports/day17_voice_reply.wav")
    reply.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    args = parser.parse_args()
    if args.command == "build-index":
        print(json.dumps(build_catalog_index(args.data_dir, args.output_dir, args.model, args.batch_size, args.limit), indent=2))
    elif args.command == "search-text":
        _print_results(search_by_text(args.query, args.top_k, args.index_dir, image_weight=args.image_weight, device=args.device))
    elif args.command == "search-image":
        _print_results(search_by_image(args.image, args.top_k, args.index_dir, image_weight=args.image_weight, device=args.device))
    elif args.command == "speak":
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        speaker = TextToSpeechSpeaker(args.model, device)
        print(json.dumps(speaker.speak(args.text, args.output), indent=2))
    else:
        knowledge_path = Path(args.knowledge)
        if not knowledge_path.is_absolute():
            knowledge_path = ROOT / knowledge_path
        if not knowledge_path.is_file():
            parser.error(f"Knowledge file not found: {knowledge_path}")
        try:
            from .day14_llm_assistant import OpenAIChatClient, RetailSupportAssistant
        except ImportError as error:
            raise RuntimeError("Could not load the Day 14 support assistant module.") from error
        support = RetailSupportAssistant(knowledge_path.read_text(encoding="utf-8"), OpenAIChatClient())
        assistant = VoiceSupportAssistant(support, TextToSpeechSpeaker(args.tts_model))
        print(json.dumps(assistant.respond(args.question, args.audio_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
