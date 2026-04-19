from argparse import ArgumentParser
import copy
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
    StableDiffusionXLControlNetInpaintPipeline,
)
from huggingface_hub import hf_hub_download


# ---------------------------------------
# Global generation settings (not CLI)
# ---------------------------------------
BASE_MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"
CONTROLNET_MODEL_ID = "xinsir/controlnet-openpose-sdxl-1.0"
OPENPOSE_ANNOTATORS_REPO = "lllyasviel/Annotators"
JUGGERNAUT_XL_DEFAULT_FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

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
INPAINT_BASE_FALLBACK_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


# ---------------------------------------
# OpenPose-face AU deltas
# ---------------------------------------
# These indices follow the common 68/70-point facial layout used by OpenPose face.
# Deltas are normalized image-space shifts.
AU_DELTAS: Dict[str, Dict[int, Tuple[float, float]]] = {
    # AU4: Brow lowerer
    "AU4": {
        17: (0.003, 0.018),
        18: (0.002, 0.020),
        19: (0.001, 0.021),
        20: (0.000, 0.020),
        21: (-0.001, 0.018),
        22: (0.001, 0.018),
        23: (0.000, 0.020),
        24: (-0.001, 0.021),
        25: (-0.002, 0.020),
        26: (-0.003, 0.018),
    },
    # AU7: Lid tightener
    "AU7": {
        37: (0.000, 0.010),
        38: (0.000, 0.011),
        40: (0.000, -0.008),
        41: (0.000, -0.008),
        43: (0.000, 0.010),
        44: (0.000, 0.011),
        46: (0.000, -0.008),
        47: (0.000, -0.008),
    },
    # AU23: Lip tightener (corners inward + lips tense)
    "AU23": {
        48: (0.010, 0.000),
        54: (-0.010, 0.000),
        51: (0.000, -0.003),
        57: (0.000, 0.003),
    },
    # AU24: Lip pressor (reduce mouth aperture)
    "AU24": {
        62: (0.000, 0.008),
        66: (0.000, -0.008),
        63: (0.000, 0.005),
        65: (0.000, -0.005),
    },
}

AU_TEXT: Dict[str, str] = {
    "AU4": "brows lowered and drawn together",
    "AU7": "eyelids tightened",
    "AU23": "lips tightened",
    "AU24": "lips pressed",
}

# OpenPose face landmark topology regions (68-point convention).
FACE_REGIONS: Dict[str, List[int]] = {
    "brows": list(range(17, 27)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
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


def load_openpose_backend() -> Tuple[Any, Any]:
    try:
        from controlnet_aux.open_pose import OpenposeDetector
        from controlnet_aux.open_pose import draw_poses
    except Exception as exc:
        raise RuntimeError(
            "OpenPose backend is required. Install with: pip install controlnet_aux opencv-python. "
            f"Original import error: {exc}"
        ) from exc

    detector = OpenposeDetector.from_pretrained(OPENPOSE_ANNOTATORS_REPO)
    if not hasattr(detector, "detect_poses"):
        raise RuntimeError(
            "Installed controlnet_aux does not provide OpenPose keypoint extraction API (detect_poses). "
            "Please install a newer controlnet_aux version."
        )
    return detector, draw_poses


def get_keypoint_xy(keypoint: Any) -> Optional[Tuple[float, float]]:
    if keypoint is None:
        return None

    if hasattr(keypoint, "x") and hasattr(keypoint, "y"):
        return float(keypoint.x), float(keypoint.y)

    if isinstance(keypoint, (list, tuple)) and len(keypoint) >= 2:
        return float(keypoint[0]), float(keypoint[1])

    return None


def clone_keypoint_with_xy(template_keypoint: Any, x: float, y: float) -> Any:
    if template_keypoint is None:
        return None

    if hasattr(template_keypoint, "_replace"):
        try:
            return template_keypoint._replace(x=float(x), y=float(y))
        except Exception:
            pass

    try:
        return type(template_keypoint)(x=float(x), y=float(y))
    except Exception:
        pass

    try:
        clone = copy.copy(template_keypoint)
        setattr(clone, "x", float(x))
        setattr(clone, "y", float(y))
        return clone
    except Exception:
        pass

    return template_keypoint


def extract_face_points(face_keypoints: List[Any]) -> List[Optional[Tuple[float, float]]]:
    points: List[Optional[Tuple[float, float]]] = []
    for kp in face_keypoints:
        points.append(get_keypoint_xy(kp))
    return points


def face_bbox_area(face_points: List[Optional[Tuple[float, float]]]) -> float:
    valid = [point for point in face_points if point is not None]
    if len(valid) < 4:
        return 0.0
    xs = [point[0] for point in valid]
    ys = [point[1] for point in valid]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def select_main_face_pose_index(poses: List[Any]) -> int:
    best_idx = -1
    best_area = -1.0
    for idx, pose in enumerate(poses):
        face_keypoints = getattr(pose, "face", None)
        if not face_keypoints:
            continue
        points = extract_face_points(face_keypoints)
        area = face_bbox_area(points)
        if area > best_area:
            best_area = area
            best_idx = idx
    if best_idx < 0:
        raise RuntimeError("OpenPose did not detect a usable face keypoint set.")
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


def inject_face_points_into_pose(pose: Any, target_face_points: List[Optional[Tuple[float, float]]]) -> Any:
    original_face = getattr(pose, "face", None)
    if original_face is None:
        raise RuntimeError("Selected pose does not contain face keypoints.")

    updated_face = []
    for idx, kp in enumerate(original_face):
        if idx >= len(target_face_points):
            updated_face.append(kp)
            continue
        target_point = target_face_points[idx]
        if target_point is None or kp is None:
            updated_face.append(kp)
            continue
        updated_face.append(clone_keypoint_with_xy(kp, target_point[0], target_point[1]))

    if hasattr(pose, "_replace"):
        return pose._replace(face=updated_face)

    clone_pose = copy.copy(pose)
    setattr(clone_pose, "face", updated_face)
    return clone_pose


def render_openpose_map(draw_poses_fn: Any, poses: List[Any], size: int) -> Image.Image:
    canvas = draw_poses_fn(
        poses,
        H=size,
        W=size,
        draw_body=False,
        draw_hand=False,
        draw_face=True,
    )
    canvas = np.asarray(canvas, dtype=np.uint8)
    if canvas.ndim != 3:
        raise RuntimeError("OpenPose draw_poses returned an unexpected canvas format.")
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


def load_pipeline(use_cuda: bool) -> StableDiffusionXLControlNetInpaintPipeline:
    dtype = torch.float16 if use_cuda else torch.float32
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL_ID, torch_dtype=dtype)

    checkpoint_path = Path(BASE_MODEL_ID).expanduser()
    if checkpoint_path.is_file():
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            checkpoint_path.as_posix(),
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    elif "::" in BASE_MODEL_ID:
        repo_id, filename = BASE_MODEL_ID.split("::", 1)
        model_path = hf_hub_download(repo_id.strip(), filename.strip())
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
    else:
        try:
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                BASE_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception:
            # Fallback to a known SDXL inpainting base in case the main checkpoint is not inpaint-compatible.
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
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

    openpose_detector, draw_poses_fn = load_openpose_backend()
    poses = openpose_detector.detect_poses(
        np.asarray(input_image, dtype=np.uint8),
        include_hand=False,
        include_face=True,
    )
    if not poses:
        raise RuntimeError("OpenPose did not detect any pose in the input image.")

    selected_idx = select_main_face_pose_index(poses)
    selected_pose = poses[selected_idx]
    original_face = extract_face_points(getattr(selected_pose, "face"))
    target_face = apply_au_deltas_to_face_points(original_face, parsed_aus)

    modified_poses = list(poses)
    modified_poses[selected_idx] = inject_face_points_into_pose(selected_pose, target_face)

    original_control_image = render_openpose_map(draw_poses_fn, poses, parsed_size)
    target_control_image = render_openpose_map(draw_poses_fn, modified_poses, parsed_size)
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
        "selected_pose_index": selected_idx,
        "base_model_id": BASE_MODEL_ID,
        "controlnet_model_id": CONTROLNET_MODEL_ID,
        "openpose_annotators_repo": OPENPOSE_ANNOTATORS_REPO,
        "steps": STEPS,
        "cfg": CFG,
        "strength": INPAINT_STRENGTH,
        "controlnet_scale": CONTROLNET_SCALE,
        "prompt": prompt,
        "negative_prompt": BASE_NEGATIVE,
        "openpose_face_original_path": original_face_path.as_posix(),
        "openpose_face_target_path": target_face_path.as_posix(),
        "control_image_original_path": control_original_path.as_posix(),
        "control_image_target_path": control_target_path.as_posix(),
        "inpaint_mask_path": mask_path.as_posix(),
        "result_image_path": result_path.as_posix(),
    }
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    print(f"Done. Result image: {result_path.as_posix()}")
    print(f"OpenPose target control map: {control_target_path.as_posix()}")
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
    parser.add_argument("--image_size", type=int, default=1024, help="Square working size for OpenPose and generation.")
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
