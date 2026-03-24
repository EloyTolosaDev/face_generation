import argparse
import itertools
import json
import multiprocessing as mp
import os
import re
import time
import torch

from expression_spec import EXPRESSIONS, ExpressionSpec
from au import ACTION_UNITS
from config import Config

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict 
from pathlib import Path
from typing import Any, Iterator, List
from compel import Compel
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm

def validate_expression_aus(expressions: List[ExpressionSpec]) -> None:
    unknown_aus = sorted({au for spec in expressions for au in spec.aus if au not in ACTION_UNITS})
    if unknown_aus:
        raise ValueError(f"Unknown AU ids referenced by expressions: {unknown_aus}")


# ---------------------------------------
# 5) Prompt templates
# ---------------------------------------
BASE_POSITIVE = (
    "photorealistic portrait photo, headshot, looking at camera, "
    "neutral studio lighting, neutral background, realistic skin texture, sharp focus, 85mm lens"
)

BASE_NEGATIVE = (
    "blurry, low quality, deformed, asymmetry, extra teeth, "
    "watermark, text, cartoon, cgi, illustration"
)


# --------------------------------------
# 6) Gender, race, age
# --------------------------------------
ETHNICITIES = ["white", "asian", "black", "hispanic", "hawaiian", "alaskan-native"]
GENDERS = ["man", "woman"]
GENDER_ETHNICS = [f"{ethnic} {gender}" for ethnic, gender in itertools.product(ETHNICITIES, GENDERS)]
AGE_GROUPS = [f"{age_range} years old" for age_range in ["15-20", "20-35", "35-50", "50-65", "65-80"]]

PARALLEL_BATCH_SIZE = 5





def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "item"


def get_combinations() -> List[tuple[str, str]]:
    return list(itertools.product(GENDER_ETHNICS, AGE_GROUPS))


def get_selected_expressions(requested_names: List[str]) -> List[ExpressionSpec]:
    validate_expression_aus(EXPRESSIONS)

    if not requested_names:
        return EXPRESSIONS

    requested = set(requested_names)
    selected = [spec for spec in EXPRESSIONS if spec.name in requested]
    if not selected:
        raise ValueError(f"No expressions matched --expressions: {requested_names}")
    return selected


# ---------------------------------------
# Phase 1: generate prompt + meta files
# ---------------------------------------
def generate_meta_files(config: Config) -> List[Path]:

    os.makedirs(config.outDir, exist_ok=True)

    selected_expressions = get_selected_expressions(config.expressions)
    combinations = get_combinations()
    total_meta = len(selected_expressions) * len(combinations) * config.seeds

    meta_paths: List[Path] = []
    base_seed = time.time_ns()
    record_index = 0

    with tqdm(total=total_meta, desc="Phase 1/2 - Writing meta", unit="meta") as pbar:
        for spec in selected_expressions:

            ## create dir where meta for each expression will live
            meta_dir = config.outDir / sanitize_token(spec.name) / "meta"
            images_dir = config.outDir / sanitize_token(spec.name) / "images"

            os.makedirs(meta_dir, exist_ok=True)

            weighted_phrases = [ACTION_UNITS[au] * strength for au, strength in spec.aus.items()]
            action_text = ", ".join(p.replace("(", "").replace(")", "") for p in weighted_phrases)

            for gender_ethnic, age_group in combinations:
                demographic_text = f"{gender_ethnic}, {age_group}"
                prompt = f"{', '.join(weighted_phrases)}, {demographic_text}, {BASE_POSITIVE}"

                for _ in range(config.seeds):
                    seed = base_seed + record_index
                    record_index += 1

                    image_stem = "_".join(
                        [
                            sanitize_token(spec.name),
                            sanitize_token(gender_ethnic),
                            sanitize_token(age_group),
                            str(seed),
                        ]
                    )
                    image_path = images_dir / f"{image_stem}.png"
                    meta_path = meta_dir / f"{image_stem}.json"

                    meta = {
                        "expression": spec.name,
                        "seed": seed,
                        "img_size": config.imgSize,
                        "steps": config.steps,
                        "cfg": config.cfg,
                        "model_id": config.modelId,
                        "au_recipe": spec.aus,
                        "au_action_text": action_text,
                        "base_positive": BASE_POSITIVE,
                        "negative_prompt": BASE_NEGATIVE,
                        "full_prompt": prompt,
                        "gender_ethnic": gender_ethnic,
                        "age_group": age_group,
                        "image_path": image_path.as_posix(),
                    }

                    with meta_path.open("w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                    meta_paths.append(meta_path)
                    pbar.update(1)

    print(f"Generated {len(meta_paths)} meta files in {config.outDir.as_posix()}.")
    return meta_paths


def get_meta_paths(from_meta: Path) -> List[Path]:
    if not from_meta.exists():
        raise FileNotFoundError(f"--from_meta path does not exist: {from_meta.as_posix()}")

    if from_meta.is_file():
        if from_meta.suffix.lower() != ".json":
            raise ValueError(f"--from_meta file must be a .json file: {from_meta.as_posix()}")
        return [from_meta]

    meta_paths = sorted(path for path in from_meta.rglob("*.json") if path.is_file())
    if not meta_paths:
        raise ValueError(f"No .json meta files found under: {from_meta.as_posix()}")
    return meta_paths


# ---------------------------------------
# Phase 2: read meta and render image
# ---------------------------------------
_PIPE: Any = None
_COMPEL: Any = None
_DEVICE: str = "cpu"


def _init_image_worker(model_id: str, use_cuda: bool, num_threads: int) -> None:
    global _PIPE, _COMPEL, _DEVICE

    _DEVICE = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    if _DEVICE == "cpu":
        torch.set_num_threads(max(1, num_threads))

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(_DEVICE)

    _PIPE = pipe
    _COMPEL = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)


def create_image(meta: dict[str, Any]) -> str:
    if _PIPE is None or _COMPEL is None:
        raise RuntimeError("Image worker is not initialized")

    prompt = meta["full_prompt"]
    negative_prompt = meta.get("negative_prompt", BASE_NEGATIVE)

    prompt_embeds = _COMPEL.build_conditioning_tensor(prompt).to(_DEVICE)
    negative_prompt_embeds = _COMPEL.build_conditioning_tensor(negative_prompt).to(_DEVICE)

    generator = torch.Generator(device=_DEVICE).manual_seed(int(meta["seed"]))

    image = _PIPE(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=int(meta["steps"]),
        guidance_scale=float(meta["cfg"]),
        generator=generator,
        width=int(meta["img_size"]),
        height=int(meta["img_size"]),
    ).images[0]

    image_path = Path(meta["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return image_path.as_posix()


def iter_batches(items: List[Path], batch_size: int) -> Iterator[List[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def render_images_from_meta(meta_paths: List[Path], config: Config) -> None:
    if not meta_paths:
        print("No meta files to render.")
        return

    ## create images subfolder 
    os.makedirs(config.outDir/"images", exist_ok=True)

    use_cuda = config.gpuOn and torch.cuda.is_available()
    if config.gpuOn and not torch.cuda.is_available():
        print("Config.gpuOn=True but CUDA is not available. Falling back to CPU.")

    if use_cuda:
        workers = 1
        print("GPU mode detected: running create_image(meta) with 1 worker to avoid GPU contention.")
    else:
        workers = max(1, min(config.parallelBatchSize, len(meta_paths)))
        print(f"CPU mode: running create_image(meta) in batches of {workers} parallel executions.")

    ctx = mp.get_context("spawn")
    total_batches = (len(meta_paths) + workers - 1) // workers
    rendered_images = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_image_worker,
        initargs=(config.modelId, use_cuda, config.numThreads),
    ) as executor:
        for batch_index, batch_paths in enumerate(iter_batches(meta_paths, workers), start=1):
            batch_metas: List[dict[str, Any]] = []
            for meta_path in batch_paths:
                with meta_path.open("r", encoding="utf-8") as f:
                    batch_metas.append(json.load(f))

            futures = {executor.submit(create_image, meta): meta for meta in batch_metas}

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Phase 2/2 - Batch {batch_index}/{total_batches}",
                unit="img",
            ):
                try:
                    future.result()
                    rendered_images += 1
                except Exception as exc:
                    failed_meta = futures[future]
                    failed_path = failed_meta.get("image_path", "unknown")
                    raise RuntimeError(f"Failed rendering image for meta target: {failed_path}") from exc

    print(f"Rendered {rendered_images} images.")


# ---------------------------------------
# Main generation pipeline
# ---------------------------------------
def main(config: Config) -> None:
    meta_paths = generate_meta_files(config) if config.fromMeta is None else get_meta_paths(config.fromMeta)

    if not config.onlyMeta: 
        render_images_from_meta(meta_paths, config)
        print(f"Done. See: {config.outDir.as_posix()}/")


if __name__ == "__main__":
    config = Config.new()

    if config.dry:
        selected = get_selected_expressions(config.expressions)
        estimated = len(selected) * len(get_combinations()) * config.seeds
        print(asdict(config))
        print({"estimated_meta_files": estimated, "estimated_images": estimated})
    else:
        main(config)
