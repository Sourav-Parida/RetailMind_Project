"""Day 16: generate catalog-image variations and assemble a short promo clip.

Example:
    python -m src.day16_image_variations_video --image data/fashion-product-images-small/images/15970.jpg \
      --product-name "Everyday Tote" --category "canvas tote" --style "minimal, warm" \
      --setting "sunlit city market" --strengths 0.25,0.45,0.65 --guidance-scales 6.5,7.5

Variation quality depends on the input image, checkpoint, runtime, and prompt.
Pixel similarity is reported as a simple diagnostic, not a measure of semantic
product fidelity or campaign quality.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .day15_marketing_imagery import DEFAULT_NEGATIVE_PROMPT, build_marketing_prompt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "day16_variations"


def load_img2img_pipeline(model_id=DEFAULT_MODEL, device=None):
    """Load the pretrained img2img pipeline once for an entire sweep."""
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
    except ImportError as error:
        raise RuntimeError("Install diffusers and PyTorch to create image variations.") from error
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    if device.startswith("cuda"):
        pipeline.enable_attention_slicing()
    return pipeline, torch, device


def pixel_fidelity(reference, candidate):
    """Return pixel MAE and PSNR after resizing the candidate to source size."""
    import numpy as np

    reference = reference.convert("RGB")
    candidate = candidate.convert("RGB").resize(reference.size, Image.Resampling.LANCZOS)
    difference = np.asarray(reference, dtype=np.float32) - np.asarray(candidate, dtype=np.float32)
    mae = float(np.abs(difference).mean() / 255)
    mse = float(np.square(difference).mean())
    psnr = None if mse == 0 else 20 * math.log10(255 / (mse ** 0.5))
    return {"pixel_mae": mae, "pixel_psnr_db": psnr}


def create_variations(
    image_path,
    product_name,
    category,
    style,
    setting,
    output_dir=DEFAULT_OUTPUT_DIR,
    strengths=(0.3, 0.5, 0.7),
    guidance_scales=(6.5, 7.5),
    seed=42,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    brand_voice="",
    model_id=DEFAULT_MODEL,
    steps=30,
    pipeline=None,
):
    """Sweep image-to-image strength/guidance and save variation benchmarks."""
    strengths = tuple(float(value) for value in strengths)
    guidance_scales = tuple(float(value) for value in guidance_scales)
    if not strengths or any(not 0.0 < value <= 1.0 for value in strengths):
        raise ValueError("Each strength must be greater than 0 and at most 1.")
    if not guidance_scales or any(not 1.0 <= value <= 20.0 for value in guidance_scales):
        raise ValueError("Each guidance scale must be between 1 and 20.")
    if not 1 <= int(steps) <= 100:
        raise ValueError("steps must be between 1 and 100.")
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Catalog image not found: {image_path}")
    source = Image.open(image_path).convert("RGB")
    # The product catalog photos are small thumbnails; upscale to a usable SD
    # size while preserving their aspect ratio and matching VAE's multiple-of-8.
    scale = min(768 / max(source.size), 512 / min(source.size))
    target_size = tuple(max(8, int(round((dimension * scale) / 8) * 8)) for dimension in source.size)
    source = source.resize(target_size, Image.Resampling.LANCZOS)
    prompt = build_marketing_prompt(
        product_name, category, style, setting, brand_voice, template="lifestyle"
    )
    if pipeline is None:
        pipeline, torch, device = load_img2img_pipeline(model_id)
    else:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to seed image variation.") from error
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    images = [source]
    for strength_index, strength in enumerate(strengths):
        for guidance_index, guidance in enumerate(guidance_scales):
            run_seed = int(seed) + strength_index * len(guidance_scales) + guidance_index
            generator = torch.Generator(device=device).manual_seed(run_seed)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=int(steps),
                generator=generator,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            variation = result.images[0]
            fidelity = pixel_fidelity(source, variation)
            output_path = output_dir / f"variation_s{strength:.2f}_g{guidance:.1f}_seed{run_seed}.png"
            variation.save(output_path)
            images.append(variation)
            results.append({
                "source_image": str(image_path),
                "product_name": product_name,
                "model_id": model_id,
                "strength": strength,
                "guidance_scale": guidance,
                "seed": run_seed,
                "steps": int(steps),
                "generation_seconds": elapsed,
                "pixel_mae": fidelity["pixel_mae"],
                "pixel_psnr_db": fidelity["pixel_psnr_db"],
                "output_path": str(output_path),
            })

    timestamp = datetime.now(timezone.utc).isoformat()
    benchmark_path = output_dir / "day16_variation_benchmark.csv"
    with benchmark_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    metadata_path = output_dir / "day16_campaign_metadata.json"
    metadata_path.write_text(json.dumps({
        "created_at_utc": timestamp,
        "positive_prompt": prompt,
        "negative_prompt": negative_prompt,
        "strengths": strengths,
        "guidance_scales": guidance_scales,
        "benchmark_note": "Pixel metrics are low-level similarity diagnostics; they do not measure semantic identity, product correctness, or visual quality.",
        "variations": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return images, results, benchmark_path, metadata_path


def assemble_promo_video(images, output_path, fps=12, transition_frames=12):
    """Crossfade a source plus selected variations into a short MP4 promo clip."""
    if len(images) < 2:
        raise ValueError("Provide the source and at least one variation to make a promo clip.")
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install imageio[ffmpeg] and numpy to assemble the MP4 clip.") from error
    frames = []
    normalized = [image.convert("RGB").resize(images[0].size, Image.Resampling.LANCZOS) for image in images]
    for before, after in zip(normalized, normalized[1:]):
        for index in range(transition_frames + 1):
            frames.append(np.asarray(Image.blend(before, after, index / transition_frames)))
        frames.extend([np.asarray(after)] * max(1, fps // 2))
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(output_path, frames, fps=fps, codec="libx264", quality=8)
    except Exception as error:
        raise RuntimeError(
            "Could not encode MP4. Install the imageio ffmpeg extra and ensure an H.264 encoder is available."
        ) from error
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Catalog product image used as img2img input")
    parser.add_argument("--product-name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--brand-voice", default="")
    parser.add_argument("--strengths", default="0.3,0.5,0.7", help="Comma-separated img2img strengths")
    parser.add_argument("--guidance-scales", default="6.5,7.5", help="Comma-separated guidance values")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--video", help="Output MP4 path (default: <output-dir>/promo_clip.mp4)")
    parser.add_argument("--no-video", action="store_true", help="Generate and benchmark variations without a video")
    args = parser.parse_args()
    strengths = [float(value) for value in args.strengths.split(",")]
    guidances = [float(value) for value in args.guidance_scales.split(",")]
    images, results, benchmark, metadata = create_variations(
        args.image, args.product_name, args.category, args.style, args.setting,
        output_dir=args.output_dir,
        strengths=strengths,
        guidance_scales=guidances,
        seed=args.seed,
        brand_voice=args.brand_voice,
        model_id=args.model,
        steps=args.steps,
    )
    print(f"Saved benchmark: {benchmark}")
    print(f"Saved campaign metadata: {metadata}")
    if not args.no_video:
        video_path = args.video or str(Path(args.output_dir) / "promo_clip.mp4")
        print(f"Saved promo clip: {assemble_promo_video(images, video_path)}")
    for result in results:
        print(f"strength={result['strength']:.2f}, guidance={result['guidance_scale']:.1f}, "
              f"seconds={result['generation_seconds']:.1f}, pixel_mae={result['pixel_mae']:.4f}")


if __name__ == "__main__":
    main()
