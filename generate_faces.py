import itertools
import json
import multiprocessing as mp
import os
import re
import time
import torch

from expression_spec import EXPRESSIONS, Expression
from config import Config

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict 
from pathlib import Path
from typing import Any, Iterator, List
from compel import Compel, ReturnedEmbeddingsType
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from tqdm import tqdm

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
AGE_GROUPS = [f"{age_range} years old" for age_range in ["15-25", "25-45", "45-65", "65-85"]]

PARALLEL_BATCH_SIZE = 5
JUGGERNAUT_XL_DEFAULT_FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "item"


def get_combinations() -> List[tuple[str, str]]:
    return list(itertools.product(GENDER_ETHNICS, AGE_GROUPS))


def get_selected_expressions(requested_names: List[str]) -> List[Expression]:

    if not requested_names:
        return EXPRESSIONS

    requested = set(requested_names)
    selected = [expression for expression in EXPRESSIONS if expression.name in requested]
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
        for expression in selected_expressions:

            ## create dir where meta for each expression will live
            meta_dir = config.outDir / sanitize_token(expression.name) / "meta"
            images_dir = config.outDir / sanitize_token(expression.name) / "images"

            os.makedirs(meta_dir, exist_ok=True)

            expression_text = [au*intensity for au, intensity in expression.config.items()]
            action_text = ", ".join(expression_text)

            for gender_ethnic, age_group in combinations:
                demographic_text = f"{gender_ethnic}, {age_group}"
                prompt = f"{', '.join(expression_text)}, {demographic_text}, {BASE_POSITIVE}"

                for _ in range(config.seeds):
                    seed = base_seed + record_index
                    record_index += 1

                    image_stem = "_".join(
                        [
                            sanitize_token(expression.name),
                            sanitize_token(gender_ethnic),
                            sanitize_token(age_group),
                            str(seed),
                        ]
                    )
                    image_path = images_dir / f"{image_stem}.png"
                    meta_path = meta_dir / f"{image_stem}.json"

                    meta = {
                        "expression": expression.name,
                        "seed": seed,
                        "img_size": config.imgSize,
                        "steps": config.steps,
                        "cfg": config.cfg,
                        "model_id": config.modelId,
                        "au_recipe": [(au.id, intensity) for au, intensity in expression.config.items()],
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


def _load_sdxl_pipe(model_id: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    checkpoint_path = Path(model_id).expanduser()
    if checkpoint_path.is_file():
        return StableDiffusionXLPipeline.from_single_file(checkpoint_path.as_posix(), torch_dtype=dtype)

    if "::" in model_id:
        repo_id, filename = model_id.split("::", 1)
        model_path = hf_hub_download(repo_id.strip(), filename.strip())
        return StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)

    try:
        return StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    except Exception:
        # Juggernaut XL is commonly distributed as a single-file checkpoint.
        model_path = hf_hub_download(model_id, JUGGERNAUT_XL_DEFAULT_FILE)
        return StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)


def _normalize_sdxl_size(size: int) -> int:
    return max(64, ((int(size) + 7) // 8) * 8)


def _init_image_worker(model_id: str, use_cuda: bool, num_threads: int) -> None:
    global _PIPE, _COMPEL, _DEVICE

    _DEVICE = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    if _DEVICE == "cpu":
        torch.set_num_threads(max(1, num_threads))

    pipe = _load_sdxl_pipe(model_id, dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    pipe = pipe.to(_DEVICE)

    _PIPE = pipe
    _COMPEL = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )


def create_image(meta: dict[str, Any]) -> str:
    if _PIPE is None or _COMPEL is None:
        raise RuntimeError("Image worker is not initialized")

    prompt = meta["full_prompt"]
    negative_prompt = meta.get("negative_prompt", BASE_NEGATIVE)

    prompt_embeds, pooled_prompt_embeds = _COMPEL(prompt)
    negative_prompt_embeds, negative_pooled_prompt_embeds = _COMPEL(negative_prompt)
    prompt_embeds, negative_prompt_embeds = _COMPEL.pad_conditioning_tensors_to_same_length(
        [prompt_embeds, negative_prompt_embeds]
    )

    generator = torch.Generator(device=_DEVICE).manual_seed(int(meta["seed"]))
    img_size = _normalize_sdxl_size(int(meta["img_size"]))

    image = _PIPE(
        prompt_embeds=prompt_embeds.to(_DEVICE),
        pooled_prompt_embeds=pooled_prompt_embeds.to(_DEVICE),
        negative_prompt_embeds=negative_prompt_embeds.to(_DEVICE),
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(_DEVICE),
        num_inference_steps=int(meta["steps"]),
        guidance_scale=float(meta["cfg"]),
        generator=generator,
        width=img_size,
        height=img_size,
    ).images[0]

    image_path = Path(meta["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return image_path.as_posix()


def iter_batches(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def resolve_image_path_from_meta(meta_path: Path, meta: dict[str, Any]) -> Path:
    """
    Build output path from the meta file location.
    Expected layout: <expression>/meta/<file>.json -> <expression>/images/<file>.png
    """
    if meta_path.parent.name.lower() == "meta":
        expression_dir = meta_path.parent.parent
    else:
        # Fallback for non-standard meta layouts.
        expression_name = sanitize_token(str(meta.get("expression", "expression")))
        expression_dir = meta_path.parent / expression_name

    return expression_dir / "images" / f"{meta_path.stem}.png"


def load_render_meta(meta_path: Path, config: Config) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # When rendering from meta, keep identity from meta (prompt/seed/expression),
    # but allow these runtime quality controls from CLI config.
    meta["img_size"] = config.imgSize
    meta["steps"] = config.steps
    meta["cfg"] = config.cfg
    meta["image_path"] = resolve_image_path_from_meta(meta_path, meta).as_posix()
    return meta


def render_images_from_meta(meta_paths: List[Path], config: Config) -> None:
    if not meta_paths:
        print("No meta files to render.")
        return

    use_cuda = config.gpuOn and torch.cuda.is_available()
    if config.gpuOn and not torch.cuda.is_available():
        print("Config.gpuOn=True but CUDA is not available. Falling back to CPU.")
    if config.imgSize % 8 != 0:
        print(
            f"--image_size {config.imgSize} is not divisible by 8. "
            f"Using {_normalize_sdxl_size(config.imgSize)} for SDXL rendering."
        )

    ctx = mp.get_context("spawn")
    rendered_images = 0

    render_metas = [load_render_meta(meta_path, config) for meta_path in meta_paths]

    metas_by_model: dict[str, List[dict[str, Any]]] = {}
    for meta in render_metas:
        model_id = str(meta.get("model_id") or config.modelId)
        metas_by_model.setdefault(model_id, []).append(meta)

    for model_id, model_metas in metas_by_model.items():
        if use_cuda:
            workers = 1
            print(f"GPU mode ({model_id}): running create_image(meta) with 1 worker to avoid GPU contention.")
        else:
            workers = max(1, min(config.parallelBatchSize, len(model_metas)))
            print(f"CPU mode ({model_id}): running create_image(meta) in batches of {workers} parallel executions.")

        total_batches = (len(model_metas) + workers - 1) // workers

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_image_worker,
            initargs=(model_id, use_cuda, config.numThreads),
        ) as executor:
            for batch_index, batch_metas in enumerate(iter_batches(model_metas, workers), start=1):
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
        print(f"Done.")


if __name__ == "__main__":
    config = Config.new()

    if config.dry:
        selected = get_selected_expressions(config.expressions)
        estimated = len(selected) * len(get_combinations()) * config.seeds
        print(asdict(config))
        print({"estimated_meta_files": estimated, "estimated_images": estimated})
    else:
        main(config)
