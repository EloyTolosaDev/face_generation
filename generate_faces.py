import itertools
import json
import multiprocessing as mp
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, List

import torch
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from au import get_au_negative_prompt
from config import Config
from expression_spec import EXPRESSIONS, Expression


# ---------------------------------------
# Prompt templates
# ---------------------------------------
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
GENDER_ETHNICS = [f"{ethnic} {gender}" for ethnic, gender in itertools.product(ETHNICITIES, GENDERS)]
AGE_GROUPS = [f"between {low} and {high} years old" for low, high in [(15, 25), (25, 45), (45, 65), (65, 85)]]

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


def iter_batches(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def compose_negative_prompt(base_negative: str, extra_negative: str) -> str:
    """Merge comma-separated negative prompt fragments with deduplication."""
    tokens: list[str] = []
    seen: set[str] = set()
    for fragment in [base_negative, extra_negative]:
        for raw_token in str(fragment or "").split(","):
            token = raw_token.strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
    return ", ".join(tokens)


def compose_neutral_prompt(demographic_text: str) -> str:
    return f"{demographic_text}, {BASE_POSITIVE}, {NEUTRAL_TEMPLATE_SUFFIX}"


def compose_expression_prompt(demographic_text: str, action_text: str) -> str:
    return f"{demographic_text}, {BASE_POSITIVE}, facial expression details: {action_text}"


def make_run_rng(base_seed: int | None, label: str) -> random.Random:
    effective_seed = int(base_seed) if base_seed is not None else time.time_ns()
    print(f"{label}: random seed generator initialized with base seed {effective_seed}")
    return random.Random(effective_seed)


def sample_seed(rng: random.Random) -> int:
    return rng.getrandbits(63)


def get_template_meta_paths(config: Config) -> List[Path]:
    source = config.templatesDir if config.templatesDir is not None else config.outDir / "neutral_templates"
    if not source.exists():
        raise FileNotFoundError(f"Template path does not exist: {source.as_posix()}")

    if source.is_file():
        if source.suffix.lower() != ".json":
            raise ValueError(f"--templates_dir file must be .json: {source.as_posix()}")
        return [source]

    candidate = source / "meta" if (source / "meta").is_dir() else source
    meta_paths = sorted(path for path in candidate.glob("*.json") if path.is_file())
    if not meta_paths:
        raise ValueError(f"No template .json files found under: {candidate.as_posix()}")
    return meta_paths


def get_meta_paths(from_meta: Path) -> List[Path]:
    if not from_meta.exists():
        raise FileNotFoundError(f"--from_meta path does not exist: {from_meta.as_posix()}")

    if from_meta.is_file():
        if from_meta.suffix.lower() != ".json":
            raise ValueError(f"--from_meta file must be .json: {from_meta.as_posix()}")
        return [from_meta]

    meta_paths = sorted(path for path in from_meta.rglob("*.json") if path.is_file())
    if not meta_paths:
        raise ValueError(f"No .json meta files found under: {from_meta.as_posix()}")
    return meta_paths


def generate_template_meta_files(config: Config) -> List[Path]:
    templates_root = config.outDir / "neutral_templates"
    meta_dir = templates_root / "meta"
    images_dir = templates_root / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    combinations = get_combinations()
    total_meta = len(combinations) * config.templateSeeds
    rng = make_run_rng(config.baseSeed, "templates")

    meta_paths: List[Path] = []

    with tqdm(total=total_meta, desc="Phase 1/2 - Writing neutral template meta", unit="meta") as pbar:
        for gender_ethnic, age_group in combinations:
            demographic_text = f"{gender_ethnic}, {age_group}"
            prompt = compose_neutral_prompt(demographic_text)

            for template_index in range(config.templateSeeds):
                seed = sample_seed(rng)
                image_stem = "_".join(
                    [
                        "neutral",
                        sanitize_token(gender_ethnic),
                        sanitize_token(age_group),
                        str(seed),
                    ]
                )
                image_path = images_dir / f"{image_stem}.png"
                meta_path = meta_dir / f"{image_stem}.json"

                meta = {
                    "kind": "neutral_template",
                    "seed": seed,
                    "template_index": template_index,
                    "img_size": config.imgSize,
                    "neutral_steps": config.neutralSteps,
                    "neutral_cfg": config.neutralCfg,
                    "model_id": config.modelId,
                    "gender_ethnic": gender_ethnic,
                    "age_group": age_group,
                    "demographic_text": demographic_text,
                    "full_prompt": prompt,
                    "negative_prompt": BASE_NEGATIVE,
                    "image_path": image_path.as_posix(),
                }

                with meta_path.open("w", encoding="utf-8") as file:
                    json.dump(meta, file, indent=2)

                meta_paths.append(meta_path)
                pbar.update(1)

    print(f"Generated {len(meta_paths)} neutral template meta files in {meta_dir.as_posix()}.")
    return meta_paths


def generate_apply_meta_files(config: Config, template_meta_paths: List[Path]) -> List[Path]:
    selected_expressions = get_selected_expressions(config.expressions)
    apply_root = config.outDir / "au_tests"

    seed_base = None if config.baseSeed is None else config.baseSeed + 1
    rng = make_run_rng(seed_base, "apply")

    total_meta = len(template_meta_paths) * len(selected_expressions) * config.applySeeds
    meta_paths: List[Path] = []

    with tqdm(total=total_meta, desc="Phase 1/2 - Writing AU apply meta", unit="meta") as pbar:
        for template_meta_path in template_meta_paths:
            with template_meta_path.open("r", encoding="utf-8") as file:
                template_meta = json.load(file)

            gender_ethnic = str(template_meta.get("gender_ethnic", "")).strip()
            age_group = str(template_meta.get("age_group", "")).strip()
            demographic_text = str(template_meta.get("demographic_text", "")).strip()
            if not demographic_text:
                demographic_text = f"{gender_ethnic}, {age_group}".strip(", ").strip()

            template_image_path = Path(str(template_meta.get("image_path", "")).strip())
            if not template_image_path.is_absolute():
                template_image_path = (Path.cwd() / template_image_path).resolve()

            template_model_id = str(template_meta.get("model_id") or config.modelId)
            template_stem = template_meta_path.stem

            for expression in selected_expressions:
                expression_token = sanitize_token(expression.name)
                expression_meta_dir = apply_root / expression_token / "meta"
                expression_images_dir = apply_root / expression_token / "images"
                expression_meta_dir.mkdir(parents=True, exist_ok=True)
                expression_images_dir.mkdir(parents=True, exist_ok=True)

                action_text = ", ".join(au.value for au in expression.aus)
                au_negative_text = get_au_negative_prompt(expression.aus)
                negative_prompt = compose_negative_prompt(BASE_NEGATIVE, au_negative_text)
                full_prompt = compose_expression_prompt(demographic_text, action_text)

                for _ in range(config.applySeeds):
                    seed = sample_seed(rng)
                    image_stem = f"{template_stem}__{expression_token}__{seed}"
                    image_path = expression_images_dir / f"{image_stem}.png"
                    meta_path = expression_meta_dir / f"{image_stem}.json"

                    meta = {
                        "kind": "au_apply",
                        "expression": expression.name,
                        "seed": seed,
                        "img_size": config.imgSize,
                        "steps": config.steps,
                        "cfg": config.cfg,
                        "strength": config.strength,
                        "model_id": template_model_id,
                        "gender_ethnic": gender_ethnic,
                        "age_group": age_group,
                        "demographic_text": demographic_text,
                        "template_meta_path": template_meta_path.as_posix(),
                        "template_image_path": template_image_path.as_posix(),
                        "au_recipe": [au.name for au in expression.aus],
                        "au_action_text": action_text,
                        "au_negative_text": au_negative_text,
                        "base_positive": BASE_POSITIVE,
                        "negative_prompt": negative_prompt,
                        "full_prompt": full_prompt,
                        "image_path": image_path.as_posix(),
                    }

                    with meta_path.open("w", encoding="utf-8") as file:
                        json.dump(meta, file, indent=2)

                    meta_paths.append(meta_path)
                    pbar.update(1)

    print(f"Generated {len(meta_paths)} AU apply meta files in {apply_root.as_posix()}.")
    return meta_paths


# ---------------------------------------
# Rendering
# ---------------------------------------
_TXT_PIPE: Any = None
_IMG_PIPE: Any = None
_COMPEL: Any = None
_DEVICE: str = "cpu"


def _load_sdxl_text2img_pipe(model_id: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
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
        model_path = hf_hub_download(model_id, JUGGERNAUT_XL_DEFAULT_FILE)
        return StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype)


def _normalize_sdxl_size(size: int) -> int:
    return max(64, ((int(size) + 7) // 8) * 8)


def _init_image_worker(model_id: str, use_cuda: bool, num_threads: int) -> None:
    global _TXT_PIPE, _IMG_PIPE, _COMPEL, _DEVICE

    _DEVICE = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    if _DEVICE == "cpu":
        torch.set_num_threads(max(1, num_threads))

    txt_pipe = _load_sdxl_text2img_pipe(model_id, dtype)
    txt_pipe.scheduler = DPMSolverMultistepScheduler.from_config(txt_pipe.scheduler.config)

    img_pipe = StableDiffusionXLImg2ImgPipeline(**txt_pipe.components)
    img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(img_pipe.scheduler.config)

    if use_cuda:
        try:
            txt_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    _TXT_PIPE = txt_pipe.to(_DEVICE)
    _IMG_PIPE = img_pipe.to(_DEVICE)

    _COMPEL = Compel(
        tokenizer=[_TXT_PIPE.tokenizer, _TXT_PIPE.tokenizer_2],
        text_encoder=[_TXT_PIPE.text_encoder, _TXT_PIPE.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
    )


def _build_conditioning(prompt: str, negative_prompt: str) -> tuple[Any, Any, Any, Any]:
    prompt_embeds, pooled_prompt_embeds = _COMPEL(prompt)
    negative_prompt_embeds, negative_pooled_prompt_embeds = _COMPEL(negative_prompt)
    prompt_embeds, negative_prompt_embeds = _COMPEL.pad_conditioning_tensors_to_same_length(
        [prompt_embeds, negative_prompt_embeds]
    )
    return (
        prompt_embeds.to(_DEVICE),
        pooled_prompt_embeds.to(_DEVICE),
        negative_prompt_embeds.to(_DEVICE),
        negative_pooled_prompt_embeds.to(_DEVICE),
    )


def _render_neutral_template(meta: dict[str, Any]) -> Any:
    prompt = str(meta["full_prompt"])
    negative_prompt = str(meta.get("negative_prompt", BASE_NEGATIVE))
    img_size = _normalize_sdxl_size(int(meta["img_size"]))

    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = _build_conditioning(prompt, negative_prompt)

    generator = torch.Generator(device=_DEVICE).manual_seed(int(meta["seed"]))
    return _TXT_PIPE(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=int(meta.get("neutral_steps", meta.get("steps", 20))),
        guidance_scale=float(meta.get("neutral_cfg", meta.get("cfg", 3.0))),
        generator=generator,
        width=img_size,
        height=img_size,
    ).images[0]


def _render_au_apply(meta: dict[str, Any]) -> Any:
    template_image_path = Path(str(meta.get("template_image_path", "")).strip())
    if not template_image_path.exists():
        raise FileNotFoundError(f"Template image not found: {template_image_path.as_posix()}")

    img_size = _normalize_sdxl_size(int(meta["img_size"]))
    template_image = Image.open(template_image_path).convert("RGB")
    if template_image.size != (img_size, img_size):
        template_image = template_image.resize((img_size, img_size), Image.Resampling.LANCZOS)

    prompt = str(meta["full_prompt"])
    negative_prompt = str(meta.get("negative_prompt", BASE_NEGATIVE))

    (
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = _build_conditioning(prompt, negative_prompt)

    generator = torch.Generator(device=_DEVICE).manual_seed(int(meta["seed"]))
    return _IMG_PIPE(
        image=template_image,
        strength=max(0.0, min(1.0, float(meta.get("strength", 0.35)))),
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=int(meta.get("steps", 24)),
        guidance_scale=float(meta.get("cfg", 5.0)),
        generator=generator,
    ).images[0]


def create_image(meta: dict[str, Any]) -> str:
    if _TXT_PIPE is None or _IMG_PIPE is None or _COMPEL is None:
        raise RuntimeError("Image worker is not initialized")

    kind = str(meta.get("kind", "")).strip().lower()
    if kind == "neutral_template":
        image = _render_neutral_template(meta)
    elif kind == "au_apply":
        image = _render_au_apply(meta)
    else:
        raise ValueError(f"Unsupported meta kind: {kind}")

    image_path = Path(meta["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return image_path.as_posix()


def resolve_image_path_from_meta(meta_path: Path, meta: dict[str, Any]) -> Path:
    if meta_path.parent.name.lower() == "meta":
        return meta_path.parent.parent / "images" / f"{meta_path.stem}.png"

    existing = str(meta.get("image_path", "")).strip()
    if existing:
        return Path(existing)

    return meta_path.with_suffix(".png")


def load_render_meta(meta_path: Path, config: Config) -> dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    kind = str(meta.get("kind", "")).strip().lower()
    if not kind:
        kind = "au_apply" if "template_image_path" in meta else "neutral_template"
    meta["kind"] = kind

    meta["model_id"] = str(meta.get("model_id") or config.modelId)
    meta["img_size"] = config.imgSize
    meta["image_path"] = resolve_image_path_from_meta(meta_path, meta).as_posix()

    if kind == "neutral_template":
        meta["negative_prompt"] = compose_negative_prompt(BASE_NEGATIVE, str(meta.get("negative_prompt", "")))
        meta["neutral_steps"] = max(1, int(config.neutralSteps))
        meta["neutral_cfg"] = float(config.neutralCfg)
        if not str(meta.get("full_prompt", "")).strip():
            demographic_text = str(meta.get("demographic_text", "")).strip()
            meta["full_prompt"] = compose_neutral_prompt(demographic_text)
        return meta

    template_image_path = Path(str(meta.get("template_image_path", "")).strip())
    if not template_image_path.is_absolute():
        template_image_path = (Path.cwd() / template_image_path).resolve()
    meta["template_image_path"] = template_image_path.as_posix()

    au_negative_text = str(meta.get("au_negative_text", "")).strip()
    merged_negative = compose_negative_prompt(BASE_NEGATIVE, str(meta.get("negative_prompt", "")).strip())
    if au_negative_text:
        merged_negative = compose_negative_prompt(merged_negative, au_negative_text)
    meta["negative_prompt"] = merged_negative

    meta["steps"] = max(1, int(config.steps))
    meta["cfg"] = float(config.cfg)
    meta["strength"] = max(0.0, min(1.0, float(config.strength)))

    if not str(meta.get("full_prompt", "")).strip():
        demographic_text = str(meta.get("demographic_text", "")).strip()
        action_text = str(meta.get("au_action_text", "")).strip()
        meta["full_prompt"] = compose_expression_prompt(demographic_text, action_text)

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
    render_metas = [load_render_meta(meta_path, config) for meta_path in meta_paths]
    metas_by_model: dict[str, List[dict[str, Any]]] = {}
    for meta in render_metas:
        metas_by_model.setdefault(str(meta["model_id"]), []).append(meta)

    rendered_images = 0

    for model_id, model_metas in metas_by_model.items():
        if use_cuda:
            workers = 1
            print(f"GPU mode ({model_id}): running with 1 worker to avoid GPU contention.")
        else:
            workers = max(1, min(config.parallelBatchSize, len(model_metas)))
            print(f"CPU mode ({model_id}): running in batches of {workers} parallel tasks.")

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


def estimate_counts(config: Config) -> dict[str, int]:
    if config.fromMeta is not None:
        meta_paths = get_meta_paths(config.fromMeta)
        return {
            "render_from_meta_files": len(meta_paths),
        }

    selected = get_selected_expressions(config.expressions)
    combinations = len(get_combinations())

    template_meta_files = 0
    apply_meta_files = 0

    if config.mode in {"templates", "full"}:
        template_meta_files = combinations * config.templateSeeds

    if config.mode == "apply":
        template_meta_count = len(get_template_meta_paths(config))
        apply_meta_files = template_meta_count * len(selected) * config.applySeeds
    elif config.mode == "full":
        apply_meta_files = template_meta_files * len(selected) * config.applySeeds

    return {
        "estimated_template_meta_files": template_meta_files,
        "estimated_apply_meta_files": apply_meta_files,
        "estimated_template_images": template_meta_files,
        "estimated_apply_images": apply_meta_files,
    }


def run_pipeline(config: Config) -> None:
    if config.fromMeta is not None:
        meta_paths = get_meta_paths(config.fromMeta)
        if config.onlyMeta:
            print(f"Loaded {len(meta_paths)} meta files from --from_meta. Skipping render due to --only_meta.")
            return
        render_images_from_meta(meta_paths, config)
        print("Done.")
        return

    if config.mode == "templates":
        template_meta_paths = generate_template_meta_files(config)
        if not config.onlyMeta:
            render_images_from_meta(template_meta_paths, config)
        print("Done.")
        return

    if config.mode == "apply":
        template_meta_paths = get_template_meta_paths(config)
        apply_meta_paths = generate_apply_meta_files(config, template_meta_paths)
        if not config.onlyMeta:
            render_images_from_meta(apply_meta_paths, config)
        print("Done.")
        return

    if config.mode == "full":
        template_meta_paths = generate_template_meta_files(config)
        if not config.onlyMeta:
            render_images_from_meta(template_meta_paths, config)

        apply_meta_paths = generate_apply_meta_files(config, template_meta_paths)
        if not config.onlyMeta:
            render_images_from_meta(apply_meta_paths, config)

        print("Done.")
        return

    raise ValueError(f"Unsupported mode: {config.mode}")


if __name__ == "__main__":
    config = Config.new()

    if config.dry:
        print(asdict(config))
        print(estimate_counts(config))
    else:
        run_pipeline(config)
