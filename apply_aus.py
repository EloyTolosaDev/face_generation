from argparse import ArgumentParser
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from huggingface_hub import hf_hub_download


# ---------------------------------------
# Global generation settings (not CLI)
# ---------------------------------------
BASE_MODEL_ID = "sd2-community/stable-diffusion-2-1"
CONTROLNET_MODEL_ID = "CrucibleAI/ControlNetMediaPipeFace"

STEPS = 24
CFG = 3.8
INPAINT_STRENGTH = 0.35
CONTROLNET_SCALE = 2.0

BASE_POSITIVE = (
    "photorealistic studio portrait photo, same person identity, same hair and skin tone, centered composition, "
    "realistic skin texture, sharp focus, neutral background, 85mm lens"
)
BASE_NEGATIVE = (
    "blurry, low quality, deformed face, asymmetry, extra teeth, watermark, text, cartoon, cgi, illustration"
)

OUTPUT_ROOT = Path("./suites/au_apply")
INPAINT_BASE_FALLBACK_MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"


# ---------------------------------------
# MediaPipe-face AU deltas
# ---------------------------------------
# Indices follow MediaPipe Face Mesh topology.
# Deltas are normalized image-space shifts.
AU_DELTAS: Dict[str, Dict[int, Tuple[float, float]]] = {
    # AU4: Brow lowerer
    "AU4": {
        70: (0.003, 0.018),
        63: (0.002, 0.020),
        105: (0.001, 0.021),
        66: (0.000, 0.020),
        107: (-0.001, 0.018),
        336: (0.001, 0.018),
        296: (0.000, 0.020),
        334: (-0.001, 0.021),
        293: (-0.002, 0.020),
        300: (-0.003, 0.018),
    },
    # AU7: Lid tightener
    "AU7": {
        159: (0.000, 0.010),
        160: (0.000, 0.011),
        158: (0.000, 0.010),
        145: (0.000, -0.008),
        153: (0.000, -0.008),
        144: (0.000, -0.008),
        386: (0.000, 0.010),
        385: (0.000, 0.011),
        387: (0.000, 0.010),
        374: (0.000, -0.008),
        380: (0.000, -0.008),
        373: (0.000, -0.008),
    },
    # AU23: Lip tightener (corners inward + lips tense)
    "AU23": {
        61: (0.010, 0.000),
        291: (-0.010, 0.000),
        13: (0.000, -0.003),
        14: (0.000, 0.003),
    },
    # AU24: Lip pressor (reduce mouth aperture)
    "AU24": {
        13: (0.000, 0.008),
        14: (0.000, -0.008),
        78: (0.000, 0.005),
        308: (0.000, -0.005),
    },
}

AU_TEXT: Dict[str, str] = {
    "AU4": "brows lowered and drawn together",
    "AU7": "eyelids tightened",
    "AU23": "lips tightened",
    "AU24": "lips pressed",
}

# MediaPipe Face Mesh regions.
FACE_REGIONS: Dict[str, List[int]] = {
    "brows": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
    "left_eye": [33, 160, 159, 158, 133, 153, 145, 144],
    "right_eye": [362, 385, 386, 387, 263, 373, 374, 380],
    "outer_mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "inner_mouth": [78, 191, 80, 13, 310, 415, 308, 324, 14, 87],
}

# Which regions should be editable per AU.
AU_TO_MASK_REGIONS: Dict[str, List[str]] = {
    "AU4": ["brows"],
    "AU7": ["left_eye", "right_eye"],
    "AU23": ["outer_mouth", "inner_mouth"],
    "AU24": ["outer_mouth", "inner_mouth"],
}


@dataclass
class ParsedAU:
    name: str
    intensity: float


def normalize_size(size: int) -> int:
    return max(64, ((int(size) + 7) // 8) * 8)


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "item"


def parse_au_token(token: str) -> ParsedAU:
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


def load_mediapipe_backend() -> Tuple[Any, Any, Any]:
    try:
        import mediapipe as mp
        from mediapipe.framework.formats import landmark_pb2
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe backend is required. Install with: pip install mediapipe. "
            f"Original import error: {exc}"
        ) from exc

    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Installed mediapipe package does not expose mediapipe.solutions. "
            "Please install a compatible mediapipe version."
        )

    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    return mp, landmark_pb2, detector


def get_landmark_xy(landmark: Any) -> Optional[Tuple[float, float]]:
    if landmark is None:
        return None
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return float(landmark.x), float(landmark.y)
    return None


def extract_face_points(face_landmarks: Any) -> List[Optional[Tuple[float, float]]]:
    points: List[Optional[Tuple[float, float]]] = []
    for lm in face_landmarks.landmark:
        points.append(get_landmark_xy(lm))
    return points


def face_bbox_area(face_points: List[Optional[Tuple[float, float]]]) -> float:
    valid = [point for point in face_points if point is not None]
    if len(valid) < 4:
        return 0.0
    xs = [point[0] for point in valid]
    ys = [point[1] for point in valid]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def select_main_face_index(face_landmark_list: List[Any]) -> int:
    best_idx = -1
    best_area = -1.0
    for idx, face_landmarks in enumerate(face_landmark_list):
        points = extract_face_points(face_landmarks)
        area = face_bbox_area(points)
        if area > best_area:
            best_area = area
            best_idx = idx
    if best_idx < 0:
        raise RuntimeError("MediaPipe did not detect a usable face landmark set.")
    return best_idx


def apply_au_deltas_to_face_points(
    face_points: List[Optional[Tuple[float, float]]],
    au_list: List[ParsedAU],
) -> List[Optional[Tuple[float, float]]]:
    target = list(face_points)
    for au in au_list:
        delta_map = AU_DELTAS[au.name]
        for idx, (dx, dy) in delta_map.items():
            if idx >= len(target):
                continue
            point = target[idx]
            if point is None:
                continue
            x = min(1.0, max(0.0, point[0] + dx * au.intensity))
            y = min(1.0, max(0.0, point[1] + dy * au.intensity))
            target[idx] = (x, y)
    return target


def inject_face_points_into_landmarks(face_landmarks: Any, target_points: List[Optional[Tuple[float, float]]], landmark_pb2: Any) -> Any:
    updated = landmark_pb2.NormalizedLandmarkList()
    for idx, lm in enumerate(face_landmarks.landmark):
        cloned = updated.landmark.add()
        cloned.x = float(lm.x)
        cloned.y = float(lm.y)
        cloned.z = float(getattr(lm, "z", 0.0))

        point = target_points[idx] if idx < len(target_points) else None
        if point is not None:
            cloned.x = float(point[0])
            cloned.y = float(point[1])
    return updated


def render_mediapipe_map(mp: Any, face_landmark_list: List[Any], size: int) -> Image.Image:
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    face_mesh = mp.solutions.face_mesh

    for face_landmarks in face_landmark_list:
        drawing_utils.draw_landmarks(
            image=canvas,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=canvas,
            landmark_list=face_landmarks,
            connections=face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        try:
            drawing_utils.draw_landmarks(
                image=canvas,
                landmark_list=face_landmarks,
                connections=face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        except Exception:
            pass
    return Image.fromarray(canvas)


def save_face_points(path: Path, face_points: List[Optional[Tuple[float, float]]]) -> None:
    data = []
    for idx, point in enumerate(face_points):
        if point is None:
            data.append({"idx": idx, "x": None, "y": None})
        else:
            data.append({"idx": idx, "x": float(point[0]), "y": float(point[1])})

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_prompt(au_list: List[ParsedAU]) -> str:
    phrases = [AU_TEXT.get(au.name, au.name) for au in au_list]
    details = ", ".join(phrases)
    return f"{BASE_POSITIVE}, facial expression details: {details}"


def build_inpaint_mask(face_points: List[Optional[Tuple[float, float]]], au_list: List[ParsedAU], size: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    active_regions: set[str] = set()
    for au in au_list:
        for region in AU_TO_MASK_REGIONS.get(au.name, []):
            active_regions.add(region)

    def to_px(point: Tuple[float, float]) -> Tuple[int, int]:
        x = int(round(point[0] * (size - 1)))
        y = int(round(point[1] * (size - 1)))
        return x, y

    point_radius = max(4, size // 96)
    for region_name in sorted(active_regions):
        indices = FACE_REGIONS.get(region_name, [])
        region_points = []
        for idx in indices:
            if idx >= len(face_points):
                continue
            point = face_points[idx]
            if point is None:
                continue
            region_points.append(to_px(point))

        if len(region_points) >= 3:
            draw.polygon(region_points, fill=255)
        elif len(region_points) == 2:
            draw.line(region_points, fill=255, width=max(4, point_radius * 2))

        for x, y in region_points:
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=255)

    # Fallback: if mask is empty, use a centered face bounding rectangle.
    if np.asarray(mask, dtype=np.uint8).max() == 0:
        valid = [point for point in face_points if point is not None]
        if valid:
            xs = [point[0] for point in valid]
            ys = [point[1] for point in valid]
            margin = 0.08
            x0 = int(max(0.0, min(xs) - margin) * (size - 1))
            y0 = int(max(0.0, min(ys) - margin) * (size - 1))
            x1 = int(min(1.0, max(xs) + margin) * (size - 1))
            y1 = int(min(1.0, max(ys) + margin) * (size - 1))
            draw.rectangle((x0, y0, x1, y1), fill=255)

    # Expand and feather mask to avoid hard seams.
    dilate_kernel = max(3, (size // 48) | 1)  # odd kernel size for MaxFilter
    feather_radius = max(2, size // 128)
    mask = mask.filter(ImageFilter.MaxFilter(dilate_kernel))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return mask


def load_pipeline(use_cuda: bool) -> StableDiffusionControlNetInpaintPipeline:
    dtype = torch.float16 if use_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=dtype)

    checkpoint_path = Path(BASE_MODEL_ID).expanduser()
    if checkpoint_path.is_file():
        pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
            checkpoint_path.as_posix(),
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    elif "::" in BASE_MODEL_ID:
        repo_id, filename = BASE_MODEL_ID.split("::", 1)
        model_path = hf_hub_download(repo_id.strip(), filename.strip())
        pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    else:
        try:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                BASE_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                INPAINT_BASE_FALLBACK_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True,
            )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe.to(device)


def run(image_path: Path, aus: List[str], image_size: int, seed: int | None, output: Path | None) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path.as_posix()}")

    parsed_aus = [parse_au_token(token) for token in aus]
    parsed_size = normalize_size(image_size)

    input_image = Image.open(image_path).convert("RGB")
    input_image = ImageOps.fit(input_image, (parsed_size, parsed_size), method=Image.Resampling.LANCZOS)

    mp, landmark_pb2, mp_face_mesh = load_mediapipe_backend()
    mp_result = mp_face_mesh.process(np.asarray(input_image, dtype=np.uint8))
    face_landmark_list = list((mp_result.multi_face_landmarks or []))
    mp_face_mesh.close()
    if not face_landmark_list:
        raise RuntimeError("MediaPipe did not detect any face landmarks in the input image.")

    selected_idx = select_main_face_index(face_landmark_list)
    selected_face_landmarks = face_landmark_list[selected_idx]
    original_face = extract_face_points(selected_face_landmarks)
    target_face = apply_au_deltas_to_face_points(original_face, parsed_aus)

    modified_face_landmark_list = list(face_landmark_list)
    modified_face_landmark_list[selected_idx] = inject_face_points_into_landmarks(
        selected_face_landmarks,
        target_face,
        landmark_pb2,
    )

    original_control_image = render_mediapipe_map(mp, face_landmark_list, parsed_size)
    target_control_image = render_mediapipe_map(mp, modified_face_landmark_list, parsed_size)
    inpaint_mask = build_inpaint_mask(original_face, parsed_aus, parsed_size)

    out_dir = output if output is not None else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    au_label = "_".join(sanitize_token(au.name) for au in parsed_aus)
    run_seed = int(seed) if seed is not None else random.getrandbits(63)
    stem = f"{sanitize_token(image_path.stem)}__{au_label}__{run_seed}"

    source_path = out_dir / f"{stem}__input.png"
    original_face_path = out_dir / f"{stem}__face_original.json"
    target_face_path = out_dir / f"{stem}__face_target.json"
    control_original_path = out_dir / f"{stem}__control_original.png"
    control_target_path = out_dir / f"{stem}__control_target.png"
    mask_path = out_dir / f"{stem}__mask.png"
    result_path = out_dir / f"{stem}__result.png"
    meta_path = out_dir / f"{stem}__meta.json"

    input_image.save(source_path)
    save_face_points(original_face_path, original_face)
    save_face_points(target_face_path, target_face)
    original_control_image.save(control_original_path)
    target_control_image.save(control_target_path)
    inpaint_mask.save(mask_path)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    pipe = load_pipeline(use_cuda=use_cuda)
    generator = torch.Generator(device=device).manual_seed(run_seed)

    prompt = build_prompt(parsed_aus)
    result = pipe(
        prompt=prompt,
        negative_prompt=BASE_NEGATIVE,
        image=input_image,
        mask_image=inpaint_mask,
        control_image=target_control_image,
        num_inference_steps=STEPS,
        guidance_scale=CFG,
        strength=INPAINT_STRENGTH,
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
        "selected_face_index": selected_idx,
        "base_model_id": BASE_MODEL_ID,
        "controlnet_model_id": CONTROLNET_MODEL_ID,
        "steps": STEPS,
        "cfg": CFG,
        "strength": INPAINT_STRENGTH,
        "controlnet_scale": CONTROLNET_SCALE,
        "prompt": prompt,
        "negative_prompt": BASE_NEGATIVE,
        "mediapipe_face_original_path": original_face_path.as_posix(),
        "mediapipe_face_target_path": target_face_path.as_posix(),
        "control_image_original_path": control_original_path.as_posix(),
        "control_image_target_path": control_target_path.as_posix(),
        "inpaint_mask_path": mask_path.as_posix(),
        "result_image_path": result_path.as_posix(),
    }
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    print(f"Done. Result image: {result_path.as_posix()}")
    print(f"MediaPipe target control map: {control_target_path.as_posix()}")
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
    parser.add_argument("--image_size", type=int, default=768, help="Square working size for MediaPipe and generation.")
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
