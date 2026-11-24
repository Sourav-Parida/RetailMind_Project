"""Day 15: reproducible, on-brand marketing images with Stable Diffusion.

Example:
    python -m src.day15_marketing_imagery --product-name "Everyday Tote" \
        --category "canvas tote bag" --style "minimal, cheerful" \
        --setting "sunlit city market" --seed 42

The Stable Diffusion weights are downloaded from Hugging Face on first use.
Check the model's license and usage terms before commercial use.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, logo, extra objects, duplicate product, distorted shape, "
    "bad proportions, blurry, low quality, oversaturated, people, hands"
)
DEFAULT_PROMPT_LIBRARY = {
    "clean_catalog": "professional catalog product photograph, clean composition, soft studio lighting, neutral background",
    "lifestyle": "premium lifestyle campaign photograph, natural lighting, authentic retail setting, polished art direction",
    "seasonal": "seasonal retail campaign photograph, inviting atmosphere, tasteful color palette, polished art direction",
}


def build_marketing_prompt(product_name, category, style, setting, brand_voice="", template="lifestyle"):
    """Build a consistent campaign prompt from product metadata."""
    fields = {
        "product name": product_name,
        "category": category,
        "style": style,
        "setting": setting,
    }
    missing = [name for name, value in fields.items() if not str(value).strip()]
    if missing:
        raise ValueError("Missing required product metadata: " + ", ".join(missing))
    if template not in DEFAULT_PROMPT_LIBRARY:
        raise ValueError(f"Unknown prompt template {template!r}; choose from {', '.join(DEFAULT_PROMPT_LIBRARY)}")
    parts = [
        f"Create a marketing image featuring {product_name}, a {category}.",
        f"Visual style: {style}.",
        f"Scene: {setting}.",
        DEFAULT_PROMPT_LIBRARY[template],
    ]
    if brand_voice.strip():
        parts.append(f"Brand direction: {brand_voice.strip()}.")
    parts.append("Show the product clearly and accurately as the hero subject; no lettering or logos.")
    return " ".join(parts)


def load_pipeline(model_id=DEFAULT_MODEL, device=None):
    """Load Stable Diffusion using diffusers and select a suitable device."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as error:
        raise RuntimeError("Install diffusers and PyTorch to generate images.") from error
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipeline = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)
    if device.startswith("cuda"):
        try:
            pipeline.enable_attention_slicing()
        except (AttributeError, RuntimeError):
            pass
    return pipeline, torch, device


def generate_marketing_image(
    product_name,
    category,
    style,
    setting,
    output_path=None,
    seed=42,
    guidance_scale=7.5,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    brand_voice="",
    template="lifestyle",
    model_id=DEFAULT_MODEL,
    steps=30,
    pipeline=None,
):
    """Generate and save a product marketing image; return image and metadata."""
    if not 1 <= int(steps) <= 100:
        raise ValueError("steps must be between 1 and 100.")
    if not 1.0 <= float(guidance_scale) <= 20.0:
        raise ValueError("guidance_scale must be between 1 and 20.")
    prompt = build_marketing_prompt(product_name, category, style, setting, brand_voice, template)
    if pipeline is None:
        pipeline, torch, device = load_pipeline(model_id)
    else:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to seed image generation.") from error
        device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    image = result.images[0]
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = "_".join(str(product_name).lower().split())[:50] or "product"
        output_path = ROOT / "reports" / "day15_marketing" / f"{safe_name}_{timestamp}_seed{seed}.png"
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    metadata = {
        "product_name": product_name,
        "category": category,
        "style": style,
        "setting": setting,
        "brand_voice": brand_voice,
        "template": template,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model_id": model_id,
        "seed": int(seed),
        "guidance_scale": float(guidance_scale),
        "steps": int(steps),
        "output_path": str(output_path),
    }
    return image, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-name", required=True, help="Product name or label")
    parser.add_argument("--category", required=True, help="Product category from catalog metadata")
    parser.add_argument("--style", required=True, help="Desired visual style, such as 'warm, minimal'")
    parser.add_argument("--setting", required=True, help="Campaign setting or scene")
    parser.add_argument("--brand-voice", default="", help="Short brand direction")
    parser.add_argument("--template", choices=DEFAULT_PROMPT_LIBRARY, default="lifestyle")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Diffusers model ID")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for repeatable generation")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", help="Output PNG path")
    args = parser.parse_args()
    _, metadata = generate_marketing_image(
        product_name=args.product_name,
        category=args.category,
        style=args.style,
        setting=args.setting,
        output_path=args.output,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        brand_voice=args.brand_voice,
        template=args.template,
        model_id=args.model,
        steps=args.steps,
    )
    metadata_path = Path(metadata["output_path"]).with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved image: {metadata['output_path']}")
    print(f"Saved generation settings: {metadata_path}")


if __name__ == "__main__":
    main()
