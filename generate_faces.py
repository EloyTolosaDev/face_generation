import argparse
import itertools
import json
import multiprocessing as mp
import os
import re
import time
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional
from compel import Compel
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm


# -----------------------------
# 2) AU -> concrete descriptions
# -----------------------------


@dataclass
class AU:
    id: int
    light_desc: str
    medium_desc: str
    strong_desc: str

    # Multiplying an AU by a float returns a weighted phrase for prompt generation.
    def __mul__(self, intensity: float) -> str:
        weight = intensity_to_weight(intensity)
        if intensity <= 0:
            return ""
        if intensity <= 0.33:
            return f"{self.light_desc}:{weight}"
        if intensity <= 0.66:
            return f"{self.medium_desc}:{weight}"
        return f"{self.strong_desc}:{weight}"


ACTION_UNITS = {
    6: AU(
        6,
        light_desc="cheeks slightly lifted",
        medium_desc="cheeks lifted, faint smile lines",
        strong_desc="cheeks strongly raised, visible smile lines, slight eye squint",
    ),
    12: AU(
        12,
        light_desc="mouth corners slightly upturned",
        medium_desc="mouth corners upturned",
        strong_desc="mouth corners strongly upturned, pronounced grin",
    ),
    25: AU(
        25,
        light_desc="lips slightly parted",
        medium_desc="lips parted",
        strong_desc="lips clearly parted",
    ),
    4: AU(
        4,
        light_desc="brows slightly drawn together",
        medium_desc="brows furrowed, eyebrows pulled down and together",
        strong_desc="brows strongly furrowed, deep crease between eyebrows",
    ),
    7: AU(
        7,
        light_desc="eyes slightly narrowed",
        medium_desc="eyes narrowed, tense eyelids",
        strong_desc="eyes tightly narrowed, intense eyelid tension",
    ),
    23: AU(
        23,
        light_desc="mouth slightly tense",
        medium_desc="lips tightened, mouth tense",
        strong_desc="lips strongly tightened, rigid tense mouth",
    ),
    24: AU(
        24,
        light_desc="lips gently pressed together",
        medium_desc="lips pressed together firmly",
        strong_desc="lips pressed tightly together, strong mouth tension",
    ),
    17: AU(
        17,
        light_desc="chin slightly raised",
        medium_desc="chin raised, lower lip pushed upward",
        strong_desc="chin strongly raised, pronounced lower-lip and chin tension",
    ),
    5: AU(
        5,
        light_desc="eyes slightly more open",
        medium_desc="upper eyelids raised, eyes more open",
        strong_desc="upper eyelids strongly raised, eyes wide open",
    ),
    10: AU(
        10,
        light_desc="upper lip slightly raised",
        medium_desc="upper lip raised, faint sneer",
        strong_desc="upper lip strongly raised (risk: drifts toward disgust)",
    ),
}


# ---------------------------------------
# 3) Simple intensity -> weight mapping
# ---------------------------------------
def intensity_to_weight(x: float) -> float:
    """
    0.0 -> 0 (omit)
    (0, 0.5] -> light -> 1.3
    (0.5, 1.0] -> strong -> 1.7
    """
    if x <= 0.0:
        return 0.0
    if x <= 0.5:
        return 1.3
    return 1.7


# ---------------------------------------
# 4) Expression definitions
# ---------------------------------------
@dataclass
class ExpressionSpec:
    name: str
    aus: dict[int, float]  # AU id -> intensity


EXPRESSIONS: List[ExpressionSpec] = [
    # ExpressionSpec("simple_smile", {12: 1.0, 6: 0.4, 25: 0.2}),
    ExpressionSpec("simple:anger", {4: 1.0, 7: 0.7, 24: 0.8, 23: 0.5}),
]


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


@dataclass
class Config:
    expressions: List[str]
    outDir: Path
    modelId: str
    imgSize: int
    steps: int
    cfg: float
    seeds: int
    gpuOn: bool
    numThreads: int
    dry: bool
    parallelBatchSize: int
    onlyMeta: bool
    fromMeta: Optional[Path]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        return cls(
            expressions=args.expressions,
            outDir=args.output,
            modelId=args.model_id,
            imgSize=args.image_size,
            steps=args.steps,
            cfg=args.cfg,
            seeds=args.seeds,
            gpuOn=args.gpu_on,
            numThreads=args.num_threads,
            dry=args.dry,
            parallelBatchSize=args.parallel_batch_size,
            onlyMeta=args.only_meta,
            fromMeta=args.from_meta,
        )


def get_config_from_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_id",
        type=str,
        default="emilianJR/epiCRealism",
        help="Model checkpoint to generate images from.",
    )
    p.add_argument("--output", type=Path, required=True, help="Output directory for generated files")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Amount of images per combination of age/ethnicity/gender/expression",
    )
    p.add_argument(
        "--gpu_on",
        default=False,
        action="store_true",
        help="Use CUDA GPU if available",
    )
    p.add_argument("--num_threads", type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument("--expressions", nargs="+", type=str, default=[])
    p.add_argument(
        "--parallel_batch_size",
        type=int,
        default=PARALLEL_BATCH_SIZE,
        help="How many create_image(meta) calls run in parallel per batch",
    )
    p.add_argument(
        "--dry",
        default=False,
        action="store_true",
        help="Dry run: print config and expected file count",
    )
    p.add_argument(
        "--only_meta",
        default=False,
        action="store_true",
        help="Only meta means only generate json files needed to generate the images. By default, meta AND images are generated"
    )
    p.add_argument("--from_meta", type=Path, help="Only generate images from meta-files from a path where meta-files are on")

    args = p.parse_args()
    return Config.from_args(args)


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "item"


def get_combinations() -> List[tuple[str, str]]:
    return list(itertools.product(GENDER_ETHNICS, AGE_GROUPS))


def get_selected_expressions(requested_names: List[str]) -> List[ExpressionSpec]:
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
    config = get_config_from_args()

    if config.dry:
        selected = get_selected_expressions(config.expressions)
        estimated = len(selected) * len(get_combinations()) * config.seeds
        print(asdict(config))
        print({"estimated_meta_files": estimated, "estimated_images": estimated})
    else:
        main(config)
