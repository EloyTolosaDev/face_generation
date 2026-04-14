from argparse import ArgumentParser
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from huggingface_hub import hf_hub_download


# ---------------------------------------
# Global generation settings (not CLI)
# ---------------------------------------
BASE_MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"
CONTROLNET_MODEL_ID = "xinsir/controlnet-openpose-sdxl-1.0"
JUGGERNAUT_XL_DEFAULT_FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

STEPS = 24
CFG = 4.5
IMG2IMG_STRENGTH = 0.35
CONTROLNET_SCALE = 1.0

BASE_POSITIVE = (
    "photorealistic studio portrait photo, same person identity, same hair and skin tone, centered composition, "
    "realistic skin texture, sharp focus, neutral background, 85mm lens"
)
BASE_NEGATIVE = (
    "blurry, low quality, deformed face, asymmetry, extra teeth, watermark, text, cartoon, cgi, illustration"
)

OUTPUT_ROOT = Path("./suites/au_apply")


# ---------------------------------------
# Landmark and AU delta configuration
# ---------------------------------------
# The deltas are in normalized image coordinates (x,y in [0,1]).
# Positive y means "move down". Negative y means "move up".
AU_DELTAS: Dict[str, Dict[int, Tuple[float, float]]] = {
    # AU4: Brow lowerer (corrugator/depressor)
    "AU4": {
        70: (0.003, 0.020),
        63: (0.002, 0.020),
        105: (0.001, 0.018),
        66: (0.000, 0.017),
        107: (-0.001, 0.017),
        336: (-0.003, 0.020),
        296: (-0.002, 0.020),
        334: (-0.001, 0.018),
        293: (0.000, 0.017),
        300: (0.001, 0.017),
    },
    # AU7: Lid tightener
    "AU7": {
        159: (0.000, 0.010),
        158: (0.000, 0.011),
        157: (0.000, 0.012),
        386: (0.000, 0.010),
        385: (0.000, 0.011),
        384: (0.000, 0.012),
        145: (0.000, -0.008),
        153: (0.000, -0.008),
        374: (0.000, -0.008),
        380: (0.000, -0.008),
    },
    # AU23: Lip tightener
    "AU23": {
        61: (0.008, 0.000),
        291: (-0.008, 0.000),
        0: (0.000, -0.004),
        17: (0.000, 0.004),
    },
    # AU24: Lip pressor
    "AU24": {
        13: (0.000, 0.010),
        14: (0.000, -0.010),
        78: (0.000, 0.006),
        308: (0.000, 0.006),
        81: (0.000, 0.006),
        311: (0.000, 0.006),
    },
}

AU_TEXT: Dict[str, str] = {
    "AU4": "brows lowered and drawn together with glabellar tension",
    "AU7": "eyelids tightened with narrowed eyes",
    "AU23": "lips tightened into a tense straight mouth",
    "AU24": "lips pressed together strongly",
}

# Feature lines used to render control map.
FEATURE_LINES: List[List[int]] = [
    [70, 63, 105, 66, 107],          # left brow
    [336, 296, 334, 293, 300],       # right brow
    [33, 160, 158, 133, 153, 144, 33],    # left eye
    [362, 385, 387, 263, 373, 380, 362],  # right eye
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],  # outer mouth
    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],   # inner mouth
    [10, 151, 9, 8, 168, 6, 197, 195, 5],  # nose bridge/tip
]


@dataclass
class ParsedAU:
    name: str
    intensity: float


def normalize_size(size: int) -> int:
    return max(64, ((int(size) + 7) // 8) * 8)


def parse_au_token(token: str) -> ParsedAU:
    """
    Supported token formats:
    - AU4
    - AU4=1.2
    - AU4_HIGH / AU4_MEDIUM / AU4_LOW
    """
    raw = str(token).strip().upper()
    if not raw:
        raise ValueError("Empty AU token")

    intensity = 1.0
    if "=" in raw:
        left, right = raw.split("=", 1)
        raw = left.strip()
        intensity = float(right.strip())
    elif raw.endswith("_HIGH"):
        raw = raw.removesuffix("_HIGH")
        intensity = 1.3
    elif raw.endswith("_MEDIUM"):
        raw = raw.removesuffix("_MEDIUM")
        intensity = 1.0
    elif raw.endswith("_LOW"):
        raw = raw.removesuffix("_LOW")
        intensity = 0.7

    if not raw.startswith("AU"):
        raw = f"AU{raw}"

    if raw not in AU_DELTAS:
        supported = ", ".join(sorted(AU_DELTAS.keys()))
        raise ValueError(f"Unsupported AU '{raw}'. Supported AUs: {supported}")

    return ParsedAU(name=raw, intensity=max(0.0, min(2.0, intensity)))


def detect_landmarks_mediapipe(image: Image.Image) -> np.ndarray:
    """
    Returns:
        numpy array of shape [468, 2] with normalized x/y coordinates.
    """
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe is required for landmark extraction. Install it with: pip install mediapipe"
        ) from exc

    rgb = np.asarray(image.convert("RGB"))
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    result = face_mesh.process(rgb)
    face_mesh.close()

    if not result.multi_face_landmarks:
        raise RuntimeError("No face detected in image.")

    pts = result.multi_face_landmarks[0].landmark
    landmarks = np.array([(pt.x, pt.y) for pt in pts], dtype=np.float32)
    return landmarks


def apply_au_deltas(landmarks: np.ndarray, au_list: List[ParsedAU]) -> np.ndarray:
    target = landmarks.copy()
    for au in au_list:
        deltas = AU_DELTAS[au.name]
        for idx, (dx, dy) in deltas.items():
            target[idx, 0] += dx * au.intensity
            target[idx, 1] += dy * au.intensity

    target[:, 0] = np.clip(target[:, 0], 0.0, 1.0)
    target[:, 1] = np.clip(target[:, 1], 0.0, 1.0)
    return target


def render_landmark_control_map(landmarks: np.ndarray, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(canvas)

    def to_px(point: Tuple[float, float]) -> Tuple[int, int]:
        x = int(round(point[0] * (size - 1)))
        y = int(round(point[1] * (size - 1)))
        return (x, y)

    for line in FEATURE_LINES:
        coords = [to_px((float(landmarks[i, 0]), float(landmarks[i, 1]))) for i in line]
        draw.line(coords, fill=(255, 255, 255), width=2)

    for idx in [70, 63, 105, 336, 296, 334, 13, 14, 61, 291, 159, 145, 386, 374]:
        px = to_px((float(landmarks[idx, 0]), float(landmarks[idx, 1])))
        r = 2
        draw.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r), fill=(255, 255, 255))

    return canvas


def build_prompt(au_list: List[ParsedAU]) -> str:
    phrases = [AU_TEXT.get(au.name, au.name) for au in au_list]
    details = ", ".join(phrases)
    return f"{BASE_POSITIVE}, facial expression details: {details}"


def load_pipeline(use_cuda: bool) -> StableDiffusionXLControlNetImg2ImgPipeline:
    dtype = torch.float16 if use_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=dtype)

    checkpoint_path = Path(BASE_MODEL_ID).expanduser()
    if checkpoint_path.is_file():
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            checkpoint_path.as_posix(),
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    elif "::" in BASE_MODEL_ID:
        repo_id, filename = BASE_MODEL_ID.split("::", 1)
        model_path = hf_hub_download(repo_id.strip(), filename.strip())
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    else:
        try:
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                BASE_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception:
            model_path = hf_hub_download(BASE_MODEL_ID, JUGGERNAUT_XL_DEFAULT_FILE)
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                model_path,
                controlnet=controlnet,
                torch_dtype=dtype,
            )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe.to(device)


def save_landmarks(path: Path, landmarks: np.ndarray) -> None:
    data = [{"idx": int(i), "x": float(pt[0]), "y": float(pt[1])} for i, pt in enumerate(landmarks)]
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "item"


def run(image_path: Path, aus: List[str], image_size: int, seed: int | None, output: Path | None) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path.as_posix()}")

    parsed_aus = [parse_au_token(token) for token in aus]
    parsed_size = normalize_size(image_size)

    input_image = Image.open(image_path).convert("RGB")
    input_image = ImageOps.fit(input_image, (parsed_size, parsed_size), method=Image.Resampling.LANCZOS)

    landmarks = detect_landmarks_mediapipe(input_image)
    target_landmarks = apply_au_deltas(landmarks, parsed_aus)
    control_image = render_landmark_control_map(target_landmarks, parsed_size)

    out_dir = output if output is not None else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    au_label = "_".join(sanitize_token(au.name) for au in parsed_aus)
    run_seed = int(seed) if seed is not None else random.getrandbits(63)
    stem = f"{sanitize_token(image_path.stem)}__{au_label}__{run_seed}"

    source_path = out_dir / f"{stem}__input.png"
    landmarks_path = out_dir / f"{stem}__landmarks_original.json"
    target_landmarks_path = out_dir / f"{stem}__landmarks_target.json"
    control_path = out_dir / f"{stem}__control.png"
    result_path = out_dir / f"{stem}__result.png"
    meta_path = out_dir / f"{stem}__meta.json"

    input_image.save(source_path)
    save_landmarks(landmarks_path, landmarks)
    save_landmarks(target_landmarks_path, target_landmarks)
    control_image.save(control_path)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    pipe = load_pipeline(use_cuda=use_cuda)
    generator = torch.Generator(device=device).manual_seed(run_seed)

    prompt = build_prompt(parsed_aus)
    result = pipe(
        prompt=prompt,
        negative_prompt=BASE_NEGATIVE,
        image=input_image,
        control_image=control_image,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        strength=IMG2IMG_STRENGTH,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        generator=generator,
    ).images[0]
    result.save(result_path)

    meta = {
        "image_path": image_path.as_posix(),
        "processed_input_path": source_path.as_posix(),
        "aus": [{"name": au.name, "intensity": au.intensity} for au in parsed_aus],
        "seed": run_seed,
        "image_size": parsed_size,
        "base_model_id": BASE_MODEL_ID,
        "controlnet_model_id": CONTROLNET_MODEL_ID,
        "steps": STEPS,
        "cfg": CFG,
        "strength": IMG2IMG_STRENGTH,
        "controlnet_scale": CONTROLNET_SCALE,
        "prompt": prompt,
        "negative_prompt": BASE_NEGATIVE,
        "landmarks_original_path": landmarks_path.as_posix(),
        "landmarks_target_path": target_landmarks_path.as_posix(),
        "control_image_path": control_path.as_posix(),
        "result_image_path": result_path.as_posix(),
    }
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    print(f"Done. Result image: {result_path.as_posix()}")
    print(f"Control map: {control_path.as_posix()}")
    print(f"Meta: {meta_path.as_posix()}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--aus",
        nargs="+",
        required=True,
        help="List of AUs (e.g. AU4 AU7 or AU4=1.2 AU7=0.8). Supported: AU4, AU7, AU23, AU24.",
    )
    parser.add_argument("--image_size", type=int, default=1024, help="Square working size for detection and generation.")
    parser.add_argument("--seed", type=int, help="Optional generation seed.")
    parser.add_argument("--output", type=Path, help="Optional output directory.")
    args = parser.parse_args()

    run(
        image_path=args.image,
        aus=args.aus,
        image_size=args.image_size,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
