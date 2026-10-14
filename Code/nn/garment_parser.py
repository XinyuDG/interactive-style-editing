import os
from collections import OrderedDict
from pathlib import Path

from PIL import Image
import numpy as np

# FASHN Human Parser labels
# 0 background
# 1 face
# 2 hair
# 3 top
# 4 dress
# 5 skirt
# 6 pants
# 7 belt
# 8 bag
# 9 hat
# 10 scarf
# 11 glasses
# 12 arms
# 13 hands
# 14 legs
# 15 feet
# 16 torso
# 17 jewelry

FASHN_LABELS = {
    "background": 0,
    "face": 1,
    "hair": 2,
    "top": 3,
    "dress": 4,
    "skirt": 5,
    "pants": 6,
    "belt": 7,
    "bag": 8,
    "hat": 9,
    "scarf": 10,
    "glasses": 11,
    "arms": 12,
    "hands": 13,
    "legs": 14,
    "feet": 15,
    "torso": 16,
    "jewelry": 17,
}

EDITABLE_REGION_SPECS = OrderedDict(
    [
        ("hair", {"label_ids": [FASHN_LABELS["hair"]], "file_stem": "hair"}),
        ("top", {"label_ids": [FASHN_LABELS["top"]], "file_stem": "top"}),
        ("dress", {"label_ids": [FASHN_LABELS["dress"]], "file_stem": "dress"}),
        ("skirt", {"label_ids": [FASHN_LABELS["skirt"]], "file_stem": "skirt"}),
        ("pants", {"label_ids": [FASHN_LABELS["pants"]], "file_stem": "pants"}),
        ("belt", {"label_ids": [FASHN_LABELS["belt"]], "file_stem": "belt"}),
        ("bag", {"label_ids": [FASHN_LABELS["bag"]], "file_stem": "bag"}),
        ("hat", {"label_ids": [FASHN_LABELS["hat"]], "file_stem": "hat"}),
        ("scarf", {"label_ids": [FASHN_LABELS["scarf"]], "file_stem": "scarf"}),
        ("glasses", {"label_ids": [FASHN_LABELS["glasses"]], "file_stem": "glasses"}),
        ("jewelry", {"label_ids": [FASHN_LABELS["jewelry"]], "file_stem": "jewelry"}),
        (
            "feet / shoes",
            {"label_ids": [FASHN_LABELS["feet"]], "file_stem": "feet_shoes"},
        ),
    ]
)

EXCLUDED_LABELS = [
    "background",
    "face",
    "arms",
    "hands",
    "legs",
    "torso",
]


def dilate_binary_mask(mask: np.ndarray, radius: int = 6) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8) * 255

    binary = mask > 0
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(binary, dtype=bool)

    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            if (dx - radius) ** 2 + (dy - radius) ** 2 <= radius**2:
                dilated |= padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]

    return dilated.astype(np.uint8) * 255


def load_label_map(label_map_path: str) -> np.ndarray:
    img = Image.open(label_map_path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def build_binary_mask(label_map: np.ndarray, target_ids: list[int]) -> np.ndarray:
    mask = np.isin(label_map, target_ids).astype(np.uint8) * 255
    return mask


def save_mask(mask: np.ndarray, save_path: str):
    Image.fromarray(mask).save(save_path)


def overlay_mask_on_image(
    image_path: str, mask: np.ndarray, save_path: str, alpha: float = 0.45
):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32)

    color = np.zeros_like(image_np)
    color[:, :, 0] = 255

    mask_bool = mask > 0
    mask_3 = np.stack([mask_bool] * 3, axis=-1)

    out = image_np.copy()
    out[mask_3] = out[mask_3] * (1 - alpha) + color[mask_3] * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(save_path)


def extract_editable_masks(label_map_path: str, image_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    label_map = load_label_map(label_map_path)
    present_label_ids = set(np.unique(label_map).astype(int).tolist())

    detected_editable_labels = [
        region_name
        for region_name, spec in EDITABLE_REGION_SPECS.items()
        if any(label_id in present_label_ids for label_id in spec["label_ids"])
    ]
    print(f"[Clothing] detected editable labels: {detected_editable_labels}")

    results = {}
    generated_mask_names = []
    for region_name, spec in EDITABLE_REGION_SPECS.items():
        if not any(label_id in present_label_ids for label_id in spec["label_ids"]):
            continue

        mask = build_binary_mask(label_map, spec["label_ids"])
        if not np.any(mask > 0):
            continue

        file_stem = spec["file_stem"]
        mask_path = os.path.join(out_dir, f"{file_stem}_mask.png")
        preview_path = os.path.join(out_dir, f"{file_stem}_preview.png")

        save_mask(mask, mask_path)
        overlay_mask_on_image(image_path, mask, preview_path)

        results[region_name] = {
            "mask_path": mask_path,
            "preview_path": preview_path,
            "label_ids": spec["label_ids"],
            "file_stem": file_stem,
        }
        generated_mask_names.append(region_name)

    print(f"[Clothing] generated editable masks: {generated_mask_names}")
    return results


def extract_background_masks(
    label_map_path: str,
    image_path: str,
    out_dir: str,
    dilation_radius: int = 6,
):
    os.makedirs(out_dir, exist_ok=True)

    label_map = load_label_map(label_map_path)
    background_mask = build_binary_mask(label_map, [FASHN_LABELS["background"]])
    person_mask = (label_map != FASHN_LABELS["background"]).astype(np.uint8) * 255
    dilated_person_mask = dilate_binary_mask(person_mask, radius=dilation_radius)
    safe_background_mask = np.where(
        (background_mask > 0) & (dilated_person_mask == 0),
        255,
        0,
    ).astype(np.uint8)

    background_mask_path = os.path.join(out_dir, "background_mask.png")
    background_preview_path = os.path.join(out_dir, "background_preview.png")
    safe_background_mask_path = os.path.join(out_dir, "safe_background_mask.png")
    safe_background_preview_path = os.path.join(out_dir, "safe_background_preview.png")
    person_mask_path = os.path.join(out_dir, "person_mask.png")
    dilated_person_mask_path = os.path.join(out_dir, "dilated_person_mask.png")

    save_mask(background_mask, background_mask_path)
    save_mask(person_mask, person_mask_path)
    save_mask(dilated_person_mask, dilated_person_mask_path)
    save_mask(safe_background_mask, safe_background_mask_path)
    overlay_mask_on_image(image_path, background_mask, background_preview_path)
    overlay_mask_on_image(
        image_path, safe_background_mask, safe_background_preview_path
    )

    return {
        "background_mask_path": background_mask_path,
        "background_preview_path": background_preview_path,
        "person_mask_path": person_mask_path,
        "dilated_person_mask_path": dilated_person_mask_path,
        "safe_background_mask_path": safe_background_mask_path,
        "safe_background_preview_path": safe_background_preview_path,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    label_map_path = project_root / "nn" / "result" / "label_map.png"
    image_path = project_root / "nn" / "picture" / "test1.jpg"
    out_dir = project_root / "nn" / "result" / "garment_masks"

    result = extract_editable_masks(
        str(label_map_path),
        str(image_path),
        str(out_dir),
    )
    print(result)
