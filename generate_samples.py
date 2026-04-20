from argparse import ArgumentParser
import itertools
import json
import os
import random
import re
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm


# ---------------------------------------
# Global generation settings
# ---------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_FALLBACK_ID = "WIBE-HuggingFace/stable-diffusion-2-1-base"
OUTPUT_ROOT = Path("./samples")
SAMPLE_IMAGE_SIZE = 768
NEUTRAL_STEPS = 20
NEUTRAL_CFG = 3.0

BASE_POSITIVE = (
    "photorealistic studio portrait photo, headshot, comfortable camera distance, entire head visible, "
    "small space above hair, centered composition, looking at camera, neutral studio lighting, "
    "neutral background, realistic skin texture, sharp focus, 85mm lens, natural facial proportions"
)

BASE_NEGATIVE = (
    "blurry, low quality, deformed, asymmetry, extra teeth, "
    "watermark, text, cartoon, cgi, illustration, extreme close-up, tightly cropped face, "
    "face filling frame, cropped forehead, cropped chin, wide-angle distortion, "
    "fisheye distortion, exaggerated facial proportions"
)

NEUTRAL_TEMPLATE_SUFFIX = (
    "neutral expression, relaxed facial muscles, mouth gently closed, neutral brow position"
)


# --------------------------------------
# Demographics
# --------------------------------------
ETHNICITIES = ["white", "asian", "black", "hispanic", "hawaiian", "alaskan-native"]
GENDERS = ["man", "woman"]
AGE_GROUPS = [f"between {low} and {high} years old" for low, high in [(15, 25), (25, 45), (45, 65), (65, 85)]]


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "item"


def get_combinations() -> list[tuple[str, str]]:
    gender_ethnics = [f"{ethnic} {gender}" for ethnic, gender in itertools.product(ETHNICITIES, GENDERS)]
    return list(itertools.product(gender_ethnics, AGE_GROUPS))


def compose_neutral_prompt(demographic_text: str) -> str:
    return f"{demographic_text}, {BASE_POSITIVE}, {NEUTRAL_TEMPLATE_SUFFIX}"


def load_text2img_pipe(use_cuda: bool) -> tuple[StableDiffusionPipeline, str]:
    dtype = torch.float16 if use_cuda else torch.float32

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    candidate_model_ids = [MODEL_ID, MODEL_FALLBACK_ID]
    last_error: Exception | None = None
    pipe: StableDiffusionPipeline | None = None
    resolved_model_id = MODEL_ID

    for model_id in candidate_model_ids:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True,
                token=token,
            )
            resolved_model_id = model_id
            break
        except Exception as exc:  # pragma: no cover - runtime dependency/network behavior
            last_error = exc

    if pipe is None:
        raise RuntimeError(
            "Could not load any configured SD2.1 sample model. "
            "If using stabilityai repos, ensure your HF account accepted the model license and set HF_TOKEN. "
            f"Last error: {last_error}"
        )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe.to(device), resolved_model_id


def normalize_image_size(size: int) -> int:
    return max(64, ((int(size) + 7) // 8) * 8)


def generate_samples(seeds_per_combination: int) -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA available: running sample generation on GPU.")
    else:
        print("CUDA unavailable: running sample generation on CPU.")

    img_size = normalize_image_size(SAMPLE_IMAGE_SIZE)

    images_dir = OUTPUT_ROOT / "images"
    meta_dir = OUTPUT_ROOT / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    combinations = get_combinations()
    total = len(combinations) * seeds_per_combination
    pipe, resolved_model_id = load_text2img_pipe(use_cuda=use_cuda)
    print(f"Using sample model: {resolved_model_id}")

    rng = random.Random(time.time_ns())
    device = "cuda" if use_cuda else "cpu"

    with tqdm(total=total, desc="Generating samples", unit="img") as pbar:
        for gender_ethnic, age_group in combinations:
            demographic_text = f"{gender_ethnic}, {age_group}"
            prompt = compose_neutral_prompt(demographic_text)

            for sample_index in range(seeds_per_combination):
                seed = rng.getrandbits(63)
                generator = torch.Generator(device=device).manual_seed(seed)
                image_stem = "_".join(
                    [
                        "sample",
                        sanitize_token(gender_ethnic),
                        sanitize_token(age_group),
                        str(seed),
                    ]
                )

                image_path = images_dir / f"{image_stem}.png"
                meta_path = meta_dir / f"{image_stem}.json"

                image = pipe(
                    prompt=prompt,
                    negative_prompt=BASE_NEGATIVE,
                    num_inference_steps=NEUTRAL_STEPS,
                    guidance_scale=NEUTRAL_CFG,
                    width=img_size,
                    height=img_size,
                    generator=generator,
                ).images[0]
                image.save(image_path)

                meta = {
                    "kind": "neutral_sample",
                    "seed": seed,
                    "sample_index": sample_index,
                    "model_id": resolved_model_id,
                    "img_size": img_size,
                    "steps": NEUTRAL_STEPS,
                    "cfg": NEUTRAL_CFG,
                    "gender_ethnic": gender_ethnic,
                    "age_group": age_group,
                    "demographic_text": demographic_text,
                    "full_prompt": prompt,
                    "negative_prompt": BASE_NEGATIVE,
                    "image_path": image_path.as_posix(),
                }
                with meta_path.open("w", encoding="utf-8") as file:
                    json.dump(meta, file, indent=2)

                pbar.update(1)

    print(f"Done. Generated {total} sample images in {images_dir.as_posix()}.")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Generate neutral sample images with txt2img.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Amount of images per demographic combination.",
    )
    args = parser.parse_args()

    if not args.samples:
        parser.error("Use --samples to run sample generation.")
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    generate_samples(seeds_per_combination=int(args.seeds))


if __name__ == "__main__":
    main()
