from argparse import ArgumentParser
import itertools
import json
import random
import re
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm


# ---------------------------------------
# Global generation settings
# ---------------------------------------
MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"
JUGGERNAUT_XL_DEFAULT_FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
OUTPUT_ROOT = Path("./samples")
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


def load_text2img_pipe(use_cuda: bool) -> StableDiffusionXLPipeline:
    dtype = torch.float16 if use_cuda else torch.float32

    checkpoint_path = Path(MODEL_ID).expanduser()
    if checkpoint_path.is_file():
        pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_path.as_posix(), torch_dtype=dtype)
    elif "::" in MODEL_ID:
        repo_id, filename = MODEL_ID.split("::", 1)
        model_path = hf_hub_download(repo_id.strip(), filename.strip())
        pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)
    else:
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception:
            model_path = hf_hub_download(MODEL_ID, JUGGERNAUT_XL_DEFAULT_FILE)
            pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe.to(device)


def normalize_sdxl_size(size: int) -> int:
    return max(64, ((int(size) + 7) // 8) * 8)


def generate_samples(seeds_per_combination: int, image_size: int) -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA available: running sample generation on GPU.")
    else:
        print("CUDA unavailable: running sample generation on CPU.")

    img_size = normalize_sdxl_size(image_size)
    if img_size != image_size:
        print(f"--image_size {image_size} adjusted to {img_size} (SDXL requires multiples of 8).")

    images_dir = OUTPUT_ROOT / "images"
    meta_dir = OUTPUT_ROOT / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    combinations = get_combinations()
    total = len(combinations) * seeds_per_combination
    pipe = load_text2img_pipe(use_cuda=use_cuda)

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
                    "model_id": MODEL_ID,
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
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Square output image size.",
    )
    args = parser.parse_args()

    if not args.samples:
        parser.error("Use --samples to run sample generation.")
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")

    generate_samples(seeds_per_combination=int(args.seeds), image_size=int(args.image_size))


if __name__ == "__main__":
    main()
