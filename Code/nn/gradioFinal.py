import os

import gc
import colorsys
import html
import inspect
import re
import socket
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter

try:
    from config import (
        AUTO_DOWNLOAD_MODELS,
        HF_HOME,
        HF_HUB_CACHE,
        LOCAL_FILES_ONLY,
        NN_DIR,
        RESULT_DIR,
    )
except ModuleNotFoundError:
    from .config import (
        AUTO_DOWNLOAD_MODELS,
        HF_HOME,
        HF_HUB_CACHE,
        LOCAL_FILES_ONLY,
        NN_DIR,
        RESULT_DIR,
    )

try:
    from garment_parser import extract_background_masks, extract_editable_masks
except ModuleNotFoundError:
    from .garment_parser import extract_background_masks, extract_editable_masks

from diffusion_model import DiffusionRefiner
from model import NeuralStyleTransfer

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_HUB_CACHE
if LOCAL_FILES_ONLY:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

GARMENT_RUNS_DIR = Path(RESULT_DIR) / "gradio_garment_runs"
GARMENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
FASHN_PARSER_CACHE_DIR = Path(HF_HUB_CACHE) / "models--fashn-ai--fashn-human-parser"

GARMENT_MASK_DISPLAY_ORDER = [
    "hair",
    "top",
    "top_refined",
    "dress",
    "skirt",
    "skirt_refined",
    "pants",
    "belt",
    "bag",
    "hat",
    "scarf",
    "glasses",
    "jewelry",
    "feet / shoes",
]

PRESET_PROMPTS = [
    "Custom (edit below)",
    "oil painting, in the style of Van Gogh, dramatic brushstrokes, vibrant colors",
    "pencil sketch, black and white, high contrast, fine detail",
    "anime style, vibrant colors, clean lines, fantasy art",
    "cyberpunk city, neon lights, high detail, volumetric lighting",
    "watercolor painting, soft colors, dreamy, atmospheric",
    "abstract expressionism, bold colors, texture, modern art",
]

SCENE_PRESETS = [
    "None",
    "Clothing / Fashion",
    "Portrait",
    "Landscape",
    "City / Street Scene",
]

NST_SCENE_PRESETS = [
    preset for preset in SCENE_PRESETS if preset != "Clothing / Fashion"
]

COLOR_TONE_PRESETS = [
    "None",
    "Warm",
    "Cool",
    "Pastel",
    "High Contrast",
]

CLOTHING_STYLE_MODES = [
    "Palette Recolor Transfer",
    "Pattern Print Transfer",
]

COLOR_SOURCE_OPTIONS = [
    "Use extracted palette color",
    "Use custom color",
]

PERSON_EDIT_TARGETS = [
    "Clothing / Accessories",
    "Background",
]

BACKGROUND_MODES = [
    "Solid Color Background",
    "Use Style Image as Background",
    "Style-Diffused Background",
]

BACKGROUND_FIT_MODES = [
    "Cover",
    "Contain",
    "Center Crop",
]

garment_parser_model = None
diffusion_refiner = None
nst_model = None
DIFFUSION_LOCK = threading.Lock()

LABEL_MAP_PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 220, 200],
        [80, 50, 20],
        [255, 0, 0],
        [255, 128, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
        [128, 0, 255],
        [255, 0, 255],
        [100, 100, 255],
        [200, 150, 150],
        [220, 180, 180],
        [180, 220, 180],
        [180, 180, 220],
        [150, 150, 150],
        [255, 215, 0],
    ],
    dtype=np.uint8,
)


def log_cuda_memory(label: str):
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(f"[GPU] {label} free/total GB = " f"{free_gb:.2f} / {total_gb:.2f}")


def get_gpu_free_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    free_bytes, _ = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def _is_private_or_vpn_ipv4(ip: str) -> bool:
    if not ip or ip in {"127.0.0.1", "0.0.0.0"}:
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(part) for part in parts]
    except ValueError:
        return False
    if any(o < 0 or o > 255 for o in octets):
        return False
    first, second = octets[0], octets[1]
    if first == 10:
        return True
    if first == 192 and second == 168:
        return True
    if first == 172 and 16 <= second <= 31:
        return True
    if first == 100 and 64 <= second <= 127:
        return True
    return False


def get_local_ip_candidates(port: int) -> List[str]:
    candidates: List[str] = []
    seen = set()

    def add_ip(ip: Optional[str]):
        ip = (ip or "").strip()
        if not _is_private_or_vpn_ipv4(ip):
            return
        if ip in seen:
            return
        seen.add(ip)
        candidates.append(f"http://{ip}:{port}")

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            add_ip(info[4][0])
    except Exception as exc:
        print(f"[Server][Warning] hostname IP lookup failed: {exc}")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            add_ip(sock.getsockname()[0])
        finally:
            sock.close()
    except Exception as exc:
        print(f"[Server][Warning] UDP IP detection failed: {exc}")

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["ipconfig"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
            )
            for match in re.findall(
                r"IPv4(?: Address| 地址)?[^\d]*(\d+\.\d+\.\d+\.\d+)",
                result.stdout,
                flags=re.IGNORECASE,
            ):
                add_ip(match)
        except Exception as exc:
            print(f"[Server][Warning] ipconfig parsing failed: {exc}")

    return candidates


def print_server_access_info(
    server_name: str,
    server_port: int,
    share: bool,
    auth_enabled: bool,
):
    print(f"[Server] server_name = {server_name}")
    print(f"[Server] server_port = {server_port}")
    print(f"[Server] share = {share}")
    print(f"[Server] auth enabled = {auth_enabled}")

    if server_name == "0.0.0.0":
        print("[Server] Gradio is listening on all network interfaces.")
        try:
            candidate_urls = get_local_ip_candidates(server_port)
        except Exception as exc:
            candidate_urls = []
            print(f"[Server][Warning] failed to detect VPN/LAN IPs: {exc}")

        if candidate_urls:
            print("[Server] Candidate VPN/LAN URLs:")
            for url in candidate_urls:
                print(f"[Server]   - {url}")
            print(
                "[Server] Share one of the above URLs with teammates on the same VPN/LAN."
            )
        else:
            print("[Server] No VPN/LAN IP detected automatically.")
            print("[Server] Run ipconfig and use the IPv4 address of your VPN adapter.")
    else:
        print("[Server] Local-only mode. Teammates cannot access this server.")
        print("[Server] To allow VPN/LAN access, set CLOTH_SERVER_NAME=0.0.0.0")

    print("[Server] If teammates cannot connect:")
    print("[Server] 1. Make sure they are on the same VPN/LAN.")
    print("[Server] 2. Make sure they use your VPN/LAN IPv4 address, not 127.0.0.1.")
    print(f"[Server] 3. Allow TCP port {server_port} through Windows Firewall.")
    print(
        "[Server] PowerShell (Admin): "
        f'New-NetFirewallRule -DisplayName "Cloth Gradio {server_port}" '
        f"-Direction Inbound -Protocol TCP -LocalPort {server_port} -Action Allow"
    )


def release_nst_memory(*objs):
    global nst_model
    print("[NST] before NST release")
    log_cuda_memory("before NST release")

    for obj in objs:
        try:
            if hasattr(obj, "model") and getattr(obj, "model", None) is not None:
                del obj.model
            del obj
        except Exception:
            pass

    if nst_model is not None:
        try:
            if hasattr(nst_model, "model") and nst_model.model is not None:
                del nst_model.model
        except Exception as exc:
            print(f"[NST][Warning] failed to delete NST model internals: {exc}")
        nst_model = None

    gc.collect()
    print("[NST] NST tensors/model released")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print("[NST] CUDA empty_cache/ipc_collect called")
    log_cuda_memory("after NST release before diffusion")


def get_nst_model():
    global nst_model
    if nst_model is None:
        nst_model = NeuralStyleTransfer()
    return nst_model


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor.detach())


def is_clothing_scene(scene_preset: Optional[str]) -> bool:
    return scene_preset == "Clothing / Fashion"


def is_person_aware_scene(scene_preset: Optional[str]) -> bool:
    return scene_preset in {"Clothing / Fashion", "Person-aware Local Edit"}


def save_label_map_preview(
    image_path: Path,
    segmentation: np.ndarray,
    preview_path: Path,
):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    color_mask = LABEL_MAP_PALETTE[segmentation]
    overlay = (0.6 * image_np + 0.4 * color_mask).astype(np.uint8)
    Image.fromarray(overlay).save(preview_path)


def get_fashn_parser():
    global garment_parser_model
    if garment_parser_model is None:
        if not FASHN_PARSER_CACHE_DIR.exists() and LOCAL_FILES_ONLY and not AUTO_DOWNLOAD_MODELS:
            raise RuntimeError(
                "Garment parser model is not cached locally. "
                "Run 'python scripts/download_models.py' first, or set "
                "CLOTH_AUTO_DOWNLOAD_MODELS=true / CLOTH_LOCAL_FILES_ONLY=false."
            )
        try:
            local_snapshot_path = None
            try:
                from huggingface_hub import snapshot_download

                local_snapshot_path = snapshot_download(
                    repo_id="fashn-ai/fashn-human-parser",
                    cache_dir=HF_HUB_CACHE,
                    local_files_only=LOCAL_FILES_ONLY and (not AUTO_DOWNLOAD_MODELS),
                    token=os.getenv("HF_TOKEN") or None,
                )
                print(
                    "[Clothing] resolved local garment parser snapshot: "
                    f"{local_snapshot_path}"
                )
            except Exception as snapshot_exc:
                print(
                    "[Clothing] snapshot_download local-only lookup failed: "
                    f"{snapshot_exc}"
                )
                snapshots_dir = FASHN_PARSER_CACHE_DIR / "snapshots"
                if snapshots_dir.is_dir():
                    snapshot_dirs = sorted(
                        [path for path in snapshots_dir.iterdir() if path.is_dir()],
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
                    if snapshot_dirs:
                        local_snapshot_path = str(snapshot_dirs[0])
                        print(
                            "[Clothing] fallback local garment parser snapshot: "
                            f"{local_snapshot_path}"
                        )

            try:
                from fashn_human_parser import FashnHumanParser
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Python package 'fashn-human-parser' is not installed. "
                    "Install dependencies from requirements.txt first."
                ) from exc

            parser_kwargs = {}
            if local_snapshot_path:
                parser_signature = inspect.signature(FashnHumanParser)
                for param_name in (
                    "pretrained_model_name_or_path",
                    "model_name_or_path",
                    "model_id",
                ):
                    if param_name in parser_signature.parameters:
                        parser_kwargs[param_name] = local_snapshot_path
                        break

            try:
                garment_parser_model = FashnHumanParser(**parser_kwargs)
            except TypeError:
                garment_parser_model = FashnHumanParser()
        except Exception as exc:
            print(f"[Clothing] failed to initialize garment parser: {exc}")
            raise RuntimeError(
                "Failed to initialize garment parser. Install 'fashn-human-parser' "
                "from requirements.txt first, and if the model is missing run "
                "'python scripts/download_models.py'."
            ) from exc
    return garment_parser_model


def get_diffusion_refiner():
    global diffusion_refiner
    if diffusion_refiner is None:
        diffusion_refiner = DiffusionRefiner()
    return diffusion_refiner


def run_diffusion_task(task_label: str, task_fn):
    print(f"[DiffusionTask] before diffusion task | {task_label}")
    log_cuda_memory(f"before diffusion task | {task_label}")
    if not DIFFUSION_LOCK.acquire(blocking=False):
        raise gr.Error(
            "Another diffusion task is running. Please wait for it to finish."
        )
    try:
        return task_fn()
    finally:
        log_cuda_memory(f"after diffusion task | {task_label}")
        print(f"[DiffusionTask] after diffusion task | {task_label}")
        DIFFUSION_LOCK.release()


def get_final_prompt(preset_choice, custom_text) -> str:
    if preset_choice == PRESET_PROMPTS[0]:
        return custom_text if custom_text else "a beautiful artwork"
    return preset_choice


def toggle_custom_prompt(preset_choice):
    is_custom = preset_choice == PRESET_PROMPTS[0]
    return gr.update(interactive=is_custom, visible=is_custom)


def try_generate_refined_masks(
    image_path: Path,
    mask_paths: Dict[str, str],
    out_dir: Path,
) -> Dict[str, str]:
    try:
        from sam2_refiner import refine_mask_with_sam2
    except Exception as exc:
        print(f"[Clothing] sam2 refiner unavailable, skip refined masks: {exc}")
        return {}

    sam2_root = Path(NN_DIR) / "sam2"
    checkpoint = sam2_root / "checkpoints" / "sam2.1_hiera_small.pt"
    if not checkpoint.is_file():
        print("[Clothing] sam2 checkpoint missing, skip refined masks")
        return {}

    refined_masks: Dict[str, str] = {}
    for mask_name, base_mask_path in mask_paths.items():
        if mask_name not in {"top", "skirt"}:
            continue

        refined_name = f"{mask_name}_refined"
        out_mask_path = out_dir / f"{mask_name}_mask_refined.png"
        out_preview_path = out_dir / f"{mask_name}_preview_refined.png"

        try:
            refine_mask_with_sam2(
                image_path=str(image_path),
                init_mask_path=base_mask_path,
                sam2_root=str(sam2_root),
                checkpoint_name="sam2.1_hiera_small.pt",
                config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
                out_mask_path=str(out_mask_path),
                out_preview_path=str(out_preview_path),
            )
            if out_mask_path.is_file():
                refined_masks[refined_name] = str(out_mask_path)
        except Exception as exc:
            print(f"[Clothing] failed to refine mask '{mask_name}': {exc}")

    return refined_masks


def run_garment_segmentation(content_img: Image.Image) -> Dict[str, object]:
    print("[PersonAware] Person-aware mode selected")
    print("[Clothing] generating garment masks for current image")
    session_dir = GARMENT_RUNS_DIR / uuid4().hex
    session_dir.mkdir(parents=True, exist_ok=True)

    image_path = session_dir / "content.png"
    label_map_path = session_dir / "label_map.png"
    label_map_preview_path = session_dir / "label_map_preview.png"
    mask_dir = session_dir / "garment_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    content_img.convert("RGB").save(image_path)

    parser = get_fashn_parser()
    segmentation = parser.predict(str(image_path))
    Image.fromarray(segmentation.astype(np.uint8)).save(label_map_path)
    save_label_map_preview(image_path, segmentation, label_map_preview_path)

    extracted = extract_editable_masks(
        label_map_path=str(label_map_path),
        image_path=str(image_path),
        out_dir=str(mask_dir),
    )
    background_masks = extract_background_masks(
        label_map_path=str(label_map_path),
        image_path=str(image_path),
        out_dir=str(mask_dir),
    )

    mask_paths: Dict[str, str] = {
        mask_name: payload["mask_path"]
        for mask_name, payload in extracted.items()
        if os.path.isfile(payload.get("mask_path", ""))
    }

    mask_paths.update(try_generate_refined_masks(image_path, mask_paths, mask_dir))

    ordered_mask_paths: Dict[str, str] = {}
    for mask_name in GARMENT_MASK_DISPLAY_ORDER:
        if mask_name in mask_paths:
            ordered_mask_paths[mask_name] = mask_paths[mask_name]

    generated_names = list(ordered_mask_paths.keys())
    print(f"[Clothing] available garment masks: {generated_names}")
    print(
        f"[PersonAware] generated background mask path: "
        f"{background_masks.get('background_mask_path')}"
    )
    print(
        f"[PersonAware] safe background mask path: "
        f"{background_masks.get('safe_background_mask_path')}"
    )

    return {
        "session_dir": str(session_dir),
        "image_path": str(image_path),
        "label_map_path": str(label_map_path),
        "label_map_preview_path": str(label_map_preview_path),
        "mask_paths": ordered_mask_paths,
        "background_mask_path": background_masks.get("background_mask_path"),
        "background_preview_path": background_masks.get("background_preview_path"),
        "safe_background_mask_path": background_masks.get("safe_background_mask_path"),
        "safe_background_preview_path": background_masks.get(
            "safe_background_preview_path"
        ),
        "person_mask_path": background_masks.get("person_mask_path"),
        "dilated_person_mask_path": background_masks.get("dilated_person_mask_path"),
        "merged_mask_path": None,
        "selected_mask_names": [],
    }


def build_mask_preview(
    image_path: str, mask_array: Optional[np.ndarray]
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if mask_array is None:
        return image

    image_np = np.array(image).astype(np.float32)
    color = np.zeros_like(image_np)
    color[:, :, 0] = 255
    mask_bool = mask_array > 0
    mask_3 = np.stack([mask_bool] * 3, axis=-1)

    out = image_np.copy()
    out[mask_3] = out[mask_3] * 0.55 + color[mask_3] * 0.45
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def load_mask_image(mask_path: str, target_size) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    return mask


def get_mask_bbox(mask_img: Image.Image):
    mask_np = np.array(mask_img)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def composite_result_with_mask(
    original_img: Image.Image,
    edited_img: Image.Image,
    mask_path: str,
) -> Image.Image:
    mask = load_mask_image(mask_path, original_img.size)
    print("[Clothing] compositing result back with selected mask")
    return Image.composite(
        edited_img.convert("RGB"),
        original_img.convert("RGB"),
        mask,
    )


def build_background_prompt(
    background_prompt: str,
    style_img_path: Optional[str],
    use_style_palette: bool,
):
    prompt = (background_prompt or "").strip()
    if not prompt:
        prompt = "replace the background with a soft artistic scene"

    if use_style_palette and style_img_path and os.path.isfile(style_img_path):
        colors = extract_style_palette(
            _load_style_reference(style_img_path), n_colors=10
        )
        if colors:
            palette_text = ", ".join(colors[:4])
            prompt = f"{prompt}, {palette_text} color palette"
    return prompt


def _fit_style_background(
    style_img: Image.Image,
    target_size,
    fit_mode: str,
) -> Image.Image:
    target_w, target_h = target_size
    src = style_img.convert("RGB")
    sw, sh = src.size
    if sw <= 0 or sh <= 0:
        return Image.new("RGB", target_size, "white")

    if fit_mode == "Contain":
        scale = min(target_w / sw, target_h / sh)
        new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
        resized = src.resize(new_size, Image.LANCZOS)
        canvas = Image.new("RGB", target_size, "white")
        x = (target_w - new_size[0]) // 2
        y = (target_h - new_size[1]) // 2
        canvas.paste(resized, (x, y))
        return canvas

    if fit_mode == "Center Crop":
        src_ratio = sw / sh
        target_ratio = target_w / target_h
        if src_ratio > target_ratio:
            new_w = int(sh * target_ratio)
            x0 = (sw - new_w) // 2
            cropped = src.crop((x0, 0, x0 + new_w, sh))
        else:
            new_h = int(sw / target_ratio)
            y0 = (sh - new_h) // 2
            cropped = src.crop((0, y0, sw, y0 + new_h))
        return cropped.resize(target_size, Image.LANCZOS)

    scale = max(target_w / sw, target_h / sh)
    new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
    resized = src.resize(new_size, Image.LANCZOS)
    x0 = max(0, (new_size[0] - target_w) // 2)
    y0 = max(0, (new_size[1] - target_h) // 2)
    return resized.crop((x0, y0, x0 + target_w, y0 + target_h))


def composite_background_with_safe_mask(
    original_img: Image.Image,
    generated_background: Image.Image,
    safe_background_mask_path: str,
    out_generated_path: Optional[str] = None,
    out_final_path: Optional[str] = None,
) -> Image.Image:
    if out_generated_path:
        generated_background.save(out_generated_path)
    result = composite_result_with_mask(
        original_img=original_img,
        edited_img=generated_background,
        mask_path=safe_background_mask_path,
    )
    if out_final_path:
        result.save(out_final_path)
    return result


def update_background_mode_ui(background_mode: str):
    is_solid = background_mode == "Solid Color Background"
    is_style_image = background_mode == "Use Style Image as Background"
    is_style_diffused = background_mode == "Style-Diffused Background"
    return (
        gr.update(visible=is_solid),
        gr.update(visible=is_style_image),
        gr.update(visible=is_style_diffused),
    )


def update_person_style_control_panels(
    edit_target: str,
    style_mode: str,
    background_mode: str,
):
    is_clothing = edit_target == "Clothing / Accessories"
    is_background = edit_target == "Background"
    is_palette_mode = is_clothing and style_mode == "Palette Recolor Transfer"
    is_pattern_mode = is_clothing and style_mode == "Pattern Print Transfer"
    is_background_solid = is_background and background_mode == "Solid Color Background"
    is_background_style_image = (
        is_background and background_mode == "Use Style Image as Background"
    )
    is_background_style_diffused = (
        is_background and background_mode == "Style-Diffused Background"
    )
    show_shared_color = is_palette_mode or is_background_solid
    show_pattern = is_clothing and style_mode == "Pattern Print Transfer"
    show_background_solid = is_background_solid
    show_background_style_image = is_background_style_image
    show_background_style_diffused = is_background_style_diffused
    show_palette_strength = is_palette_mode
    show_background_solid = False
    show_step4_group = any(
        [
            show_shared_color,
            show_pattern,
            show_background_style_image,
            show_background_style_diffused,
        ]
    )
    return (
        gr.update(visible=show_step4_group),
        gr.update(visible=show_shared_color),
        gr.update(visible=show_pattern),
        gr.update(visible=show_background_solid),
        gr.update(visible=show_background_style_image),
        gr.update(visible=show_background_style_diffused),
        gr.update(visible=show_palette_strength),
    )


def update_person_edit_mode_ui(
    edit_target: str,
    style_mode: str,
    background_mode: str,
):
    # outputs:
    # 0 person_step4_group
    # 1 person_palette_controls_group
    # 2 person_pattern_controls_group
    # 3 person_background_solid_controls_group
    # 4 person_background_style_image_panel
    # 5 person_background_style_diffused_panel
    # 6 person_palette_strength_group
    return update_person_style_control_panels(
        edit_target=edit_target,
        style_mode=style_mode,
        background_mode=background_mode,
    )


def apply_background_repaint(
    content_img: Image.Image,
    style_img_path: Optional[str],
    background_mode: str,
    background_prompt: str,
    use_style_palette: bool,
    color_source: str,
    selected_palette_color: Optional[str],
    custom_color: str,
    palette_colors: List[str],
    background_strength: float,
    steps: int,
    guidance: float,
    background_additional_instruction: str,
    background_fit_mode: str,
    background_blend_strength: float,
    garment_mask_state: Dict[str, object],
    progress=gr.Progress(),
):
    if content_img is None:
        raise gr.Error("Please upload a content image first.")
    if not garment_mask_state or not garment_mask_state.get(
        "safe_background_mask_path"
    ):
        raise gr.Error("Please click Step 1: Detect Person Regions first.")

    safe_background_mask_path = garment_mask_state.get("safe_background_mask_path")
    if not safe_background_mask_path or not os.path.isfile(safe_background_mask_path):
        raise gr.Error("Safe background mask is not available for current image.")

    print("[PersonAware] edit target: background")
    print(
        f"[PersonAware] generated background mask path: {garment_mask_state.get('background_mask_path')}"
    )
    print(f"[PersonAware] safe background mask path: {safe_background_mask_path}")
    print("[PersonAware] protected person region preserved")
    background_strength = safe_float(
        background_strength,
        0.45,
        "background_img2img_strength",
    )
    steps = safe_int(steps, 20, "background_steps")
    guidance = safe_float(guidance, 7.0, "background_guidance")
    background_blend_strength = safe_float(
        background_blend_strength,
        1.0,
        "background_blend_strength",
    )
    original = content_img.convert("RGB")
    session_dir = Path(garment_mask_state["session_dir"])
    generated_background_path = session_dir / "generated_background.png"
    final_background_composite_path = session_dir / "final_background_composite.png"

    if background_mode == "Solid Color Background":
        active_color = get_active_selected_color(
            color_source=color_source,
            selected_palette_color=selected_palette_color,
            custom_color=custom_color,
            palette_colors=palette_colors,
        )
        rgb = tuple(parse_hex_color(active_color).astype(np.uint8).tolist())
        generated_background = Image.new("RGB", original.size, rgb)
        print("[PersonAware] background mode selected: Solid Color Background")
        result = composite_background_with_safe_mask(
            original_img=original,
            generated_background=generated_background,
            safe_background_mask_path=safe_background_mask_path,
            out_generated_path=str(generated_background_path),
            out_final_path=str(final_background_composite_path),
        )
        print("[PersonAware] final composite with background mask")
        return result

    if background_mode == "Use Style Image as Background":
        if not style_img_path or not os.path.isfile(style_img_path):
            raise gr.Error("Please upload a Style Image first.")
        style_img = _load_style_reference(style_img_path)
        fitted_background = _fit_style_background(
            style_img=style_img,
            target_size=original.size,
            fit_mode=background_fit_mode,
        )
        blend_strength = float(np.clip(background_blend_strength, 0.0, 1.0))
        generated_background = (
            Image.blend(original, fitted_background, blend_strength)
            if blend_strength < 1.0
            else fitted_background
        )
        print("[PersonAware] background mode selected: Use Style Image as Background")
        result = composite_background_with_safe_mask(
            original_img=original,
            generated_background=generated_background,
            safe_background_mask_path=safe_background_mask_path,
            out_generated_path=str(generated_background_path),
            out_final_path=str(final_background_composite_path),
        )
        print("[PersonAware] final composite with background mask")
        return result

    prompt = build_background_prompt(
        background_prompt=background_prompt,
        style_img_path=style_img_path,
        use_style_palette=use_style_palette,
    )
    if background_additional_instruction and background_additional_instruction.strip():
        prompt = f"{prompt}, {background_additional_instruction.strip()}"
    print("[PersonAware] background mode selected: Style-Diffused Background")
    print("[PersonAware] generating full background from style image with img2img")
    print(f"[PersonAware] background prompt: {prompt}")
    if not style_img_path or not os.path.isfile(style_img_path):
        raise gr.Error("Please upload a Style Image first.")

    style_img = _load_style_reference(style_img_path)
    style_init = _fit_style_background(style_img, (512, 512), "Cover")
    log_cuda_memory("before background img2img inference")

    def background_progress(p):
        progress(p, desc="Generating full background...")

    refiner = get_diffusion_refiner()
    refiner.unload_inpaint_pipe()
    refiner.clear_cuda_memory()
    try:
        generated_background_512 = run_diffusion_task(
            "background img2img",
            lambda: refiner.refine(
                init_image=style_init,
                prompt=prompt,
                strength=float(np.clip(background_strength, 0.2, 0.7)),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                keep_resolution=False,
                scene_preset="Background",
                color_tone=None,
                style_image_path=None,
                blend_style_in_roi=False,
                garment_mask_path=None,
                edit_category=None,
                progress_callback=background_progress,
            ),
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise gr.Error(
            "GPU memory is not enough for this diffusion task. Try lower steps, lower strength, or restart Gradio."
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            raise gr.Error(
                "GPU memory is not enough for this diffusion task. Try lower steps, lower strength, or restart Gradio."
            ) from exc
        raise gr.Error(message) from exc
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise gr.Error(f"Background style diffusion failed: {exc}") from exc

    log_cuda_memory("after background img2img inference")
    refiner.unload_img2img_pipe()
    refiner.clear_cuda_memory()
    generated_background = _fit_style_background(
        generated_background_512,
        original.size,
        "Cover",
    )
    print(f"[PersonAware] generated background size: {generated_background.size}")
    print("[PersonAware] compositing generated background behind protected person")
    result = composite_background_with_safe_mask(
        original_img=original,
        generated_background=generated_background,
        safe_background_mask_path=safe_background_mask_path,
        out_generated_path=str(generated_background_path),
        out_final_path=str(final_background_composite_path),
    )
    print("[PersonAware] final composite with background mask")
    return result


def rgb_to_hex(rgb) -> str:
    r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def parse_hex_color(hex_color: str):
    value = (hex_color or "").strip()
    if not value:
        raise gr.Error("Please choose a theme color or enter a custom color.")
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise gr.Error("Custom Color must use #RRGGBB format.")
    try:
        return np.array(
            [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise gr.Error("Custom Color must use valid hex digits.") from exc


def extract_style_palette(style_img, n_colors=10):
    style_rgb = style_img.convert("RGB").resize((192, 192), Image.LANCZOS)
    arr = np.array(style_rgb)
    pixels = arr.reshape(-1, 3).astype(np.uint8)

    def rgb_to_hsv_tuple(pixel):
        r, g, b = pixel.astype(np.float32) / 255.0
        return colorsys.rgb_to_hsv(r, g, b)

    def is_useful_color(pixel):
        _, s, v = rgb_to_hsv_tuple(pixel)
        if v < 0.08:
            return False
        if v > 0.985 and s < 0.10:
            return False
        if s < 0.06 and 0.12 < v < 0.92:
            return False
        return True

    filtered_pixels = np.array(
        [pixel for pixel in pixels if is_useful_color(pixel)],
        dtype=np.uint8,
    )
    if len(filtered_pixels) < max(64, n_colors * 20):
        filtered_pixels = pixels

    palette_source = Image.fromarray(filtered_pixels.reshape((-1, 1, 3)), mode="RGB")
    quantized = palette_source.quantize(
        colors=max(24, n_colors * 3), method=Image.Quantize.MEDIANCUT
    )
    palette_data = quantized.getpalette()
    color_counts = quantized.getcolors() or []

    candidates = []
    total_count = sum(count for count, _ in color_counts) or 1
    for count, palette_index in color_counts:
        base = palette_index * 3
        rgb = np.array(palette_data[base : base + 3], dtype=np.uint8)
        if len(rgb) != 3:
            continue

        h, s, v = rgb_to_hsv_tuple(rgb)
        if v < 0.06:
            continue
        if v > 0.99 and s < 0.08:
            continue

        count_ratio = count / total_count
        score = count_ratio * (0.55 + s) * (0.45 + v)
        accent_score = (0.20 + count_ratio) * (0.35 + s * 1.4) * (0.40 + v * 1.1)
        candidates.append(
            {
                "rgb": rgb,
                "hex": rgb_to_hex(rgb),
                "count": count,
                "count_ratio": count_ratio,
                "h": h,
                "s": s,
                "v": v,
                "score": score,
                "accent_score": accent_score,
            }
        )

    def is_similar(candidate, chosen):
        hue_gap = min(
            abs(candidate["h"] - chosen["h"]), 1.0 - abs(candidate["h"] - chosen["h"])
        )
        sat_gap = abs(candidate["s"] - chosen["s"])
        val_gap = abs(candidate["v"] - chosen["v"])
        rgb_gap = (
            np.mean(
                np.abs(
                    candidate["rgb"].astype(np.int16) - chosen["rgb"].astype(np.int16)
                )
            )
            / 255.0
        )
        return hue_gap < 0.055 and sat_gap < 0.18 and val_gap < 0.18 and rgb_gap < 0.16

    def select_diverse_colors(sorted_candidates, target_count):
        chosen = []
        for candidate in sorted_candidates:
            if any(is_similar(candidate, existing) for existing in chosen):
                continue
            chosen.append(candidate)
            if len(chosen) >= target_count:
                break
        return chosen

    dominant_candidates = sorted(
        candidates, key=lambda item: item["score"], reverse=True
    )
    accent_candidates = sorted(
        [item for item in candidates if item["s"] >= 0.20 and item["v"] >= 0.18],
        key=lambda item: item["accent_score"],
        reverse=True,
    )

    dominant = select_diverse_colors(dominant_candidates, max(4, n_colors // 2))
    accent = select_diverse_colors(accent_candidates, max(4, n_colors // 2))

    final_candidates = []
    for candidate in accent + dominant:
        if any(existing["hex"] == candidate["hex"] for existing in final_candidates):
            continue
        if any(is_similar(candidate, existing) for existing in final_candidates):
            continue
        final_candidates.append(candidate)
        if len(final_candidates) >= n_colors:
            break

    if not final_candidates and candidates:
        final_candidates = select_diverse_colors(dominant_candidates, n_colors)

    if len(final_candidates) < n_colors:
        relaxed_candidates = sorted(
            candidates,
            key=lambda item: (item["accent_score"], item["score"]),
            reverse=True,
        )
        for candidate in relaxed_candidates:
            if any(
                existing["hex"] == candidate["hex"] for existing in final_candidates
            ):
                continue
            final_candidates.append(candidate)
            if len(final_candidates) >= n_colors:
                break

    if not final_candidates:
        avg_rgb = tuple(np.mean(filtered_pixels, axis=0).astype(np.uint8).tolist())
        final_candidates = [{"hex": rgb_to_hex(avg_rgb), "s": 0.0, "v": 0.0}]
        dominant = final_candidates
        accent = []

    if len(final_candidates) < n_colors and final_candidates:
        seed_colors = [candidate["hex"] for candidate in final_candidates]
        idx = 0
        while len(final_candidates) < n_colors:
            base_hex = seed_colors[idx % len(seed_colors)]
            base_rgb = parse_hex_color(base_hex)
            jitter = np.array(
                [
                    6 * ((idx % 3) - 1),
                    5 * (((idx + 1) % 3) - 1),
                    4 * (((idx + 2) % 3) - 1),
                ],
                dtype=np.float32,
            )
            varied_rgb = np.clip(base_rgb + jitter, 0, 255)
            varied_hex = rgb_to_hex(varied_rgb)
            if not any(existing["hex"] == varied_hex for existing in final_candidates):
                final_candidates.append({"hex": varied_hex, "s": 0.0, "v": 0.0})
            idx += 1

    dominant_hex = [candidate["hex"] for candidate in dominant]
    accent_hex = [candidate["hex"] for candidate in accent]
    final_hex = [candidate["hex"] for candidate in final_candidates[:n_colors]]
    print(f"[Clothing] dominant candidates count: {len(dominant_candidates)}")
    print(f"[Clothing] accent candidates count: {len(accent_candidates)}")
    print(f"[Clothing] final palette length: {len(final_hex)}")
    print(f"[Clothing] extracted dominant colors: {dominant_hex}")
    print(f"[Clothing] extracted accent colors: {accent_hex}")
    print(f"[Clothing] final palette colors: {final_hex}")
    default_color = accent_hex[0] if accent_hex else (final_hex[0] if final_hex else "")
    print(f"[Clothing] default selected color: {default_color}")
    if len(final_hex) != n_colors:
        print(f"[Clothing][Warning] final palette length != {n_colors}")
    return final_hex


def save_palette_preview(colors: List[str], save_path: str):
    swatch_w = 100
    swatch_h = 60
    preview = Image.new("RGB", (max(1, len(colors)) * swatch_w, swatch_h), "white")
    for idx, hex_color in enumerate(colors):
        rgb = tuple(parse_hex_color(hex_color).astype(np.uint8).tolist())
        swatch = Image.new("RGB", (swatch_w, swatch_h), rgb)
        preview.paste(swatch, (idx * swatch_w, 0))
    preview.save(save_path)


def _is_light_color(rgb) -> bool:
    return float(np.mean(rgb)) > 180.0


def build_palette_gallery_items(
    colors: List[str],
    selected_color: Optional[str] = None,
):
    items = []
    normalized_colors = list(colors or [])
    if normalized_colors:
        idx = 0
        while len(normalized_colors) < 10:
            normalized_colors.append(normalized_colors[idx % len(normalized_colors)])
            idx += 1
    for hex_color in normalized_colors[:10]:
        rgb = tuple(parse_hex_color(hex_color).astype(np.uint8).tolist())
        tile = Image.new("RGB", (160, 160), rgb)
        draw = ImageDraw.Draw(tile)
        border_color = (255, 96, 32) if hex_color == selected_color else (255, 255, 255)
        border_width = 6 if hex_color == selected_color else 2
        for offset in range(border_width):
            draw.rectangle(
                (
                    offset,
                    offset,
                    tile.width - 1 - offset,
                    tile.height - 1 - offset,
                ),
                outline=border_color,
            )

        items.append(tile)
    return items


def build_color_preview_html(hex_color: Optional[str]):
    if not hex_color:
        return (
            '<div class="custom-color-preview">'
            '<span class="custom-color-preview-swatch" '
            'style="background:#ffffff;"></span>'
            "</div>"
        )

    color = (hex_color or "").strip()
    try:
        parse_hex_color(color)
    except Exception:
        return (
            '<div class="custom-color-preview">'
            '<span class="custom-color-preview-swatch" '
            'style="background:#ffffff;"></span>'
            "</div>"
        )

    return (
        '<div class="custom-color-preview">'
        f'<span class="custom-color-preview-swatch" '
        f'style="background:{color};"></span>'
        "</div>"
    )


def build_empty_state_html(text: str, min_height: int = 320):
    safe_text = html.escape(text or "")
    return (
        f'<div class="empty-state-panel" style="min-height:{int(min_height)}px;">'
        f'<div class="empty-state-copy">{safe_text}</div>'
        "</div>"
    )


RELAYOUT_JS = """
(...args) => {
  setTimeout(() => window.dispatchEvent(new Event("resize")), 50);
  setTimeout(() => window.dispatchEvent(new Event("resize")), 250);
  return args;
}
"""


def log_callback_return(name: str, values):
    if not isinstance(values, (list, tuple)):
        values = [values]
    print(f"[UI] {name} outputs count = {len(values)}")
    for idx, value in enumerate(values):
        print(f"[UI] {name} return[{idx}] type = {type(value)}")


def image_placeholder_updates(
    image_value,
    empty_text: str,
    min_height: int = 320,
):
    return show_image_or_empty(
        image=image_value,
        empty_text=empty_text,
        min_height=min_height,
    )


def show_image_or_empty(image, empty_text: str, min_height: int = 320):
    if image is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(
                value=build_empty_state_html(empty_text, min_height=min_height),
                visible=True,
            ),
        )
    return (
        gr.update(value=image, visible=True),
        gr.update(visible=False),
    )


def palette_placeholder_updates(palette_items: Optional[List[Image.Image]]):
    has_items = bool(palette_items)
    return (
        gr.update(value=palette_items or [], visible=has_items),
        gr.update(value="", visible=False),
    )


def _load_style_reference(style_img_path: str) -> Image.Image:
    if not style_img_path or not os.path.isfile(style_img_path):
        raise gr.Error("Please upload a Style Image for Clothing / Fashion mode.")
    return Image.open(style_img_path).convert("RGB")


def _center_square_crop(image: Image.Image) -> Image.Image:
    sw, sh = image.size
    side = min(sw, sh)
    sx0 = (sw - side) // 2
    sy0 = (sh - side) // 2
    return image.crop((sx0, sy0, sx0 + side, sy0 + side))


def _tile_style_to_size(style_img: Image.Image, target_size, tile_size: int):
    tile_side = max(1, int(tile_size))
    tile = _center_square_crop(style_img).resize((tile_side, tile_side), Image.LANCZOS)
    target_w, target_h = target_size
    tiled = Image.new("RGB", target_size)
    for x in range(0, target_w, tile.width):
        for y in range(0, target_h, tile.height):
            tiled.paste(tile, (x, y))
    return tiled.crop((0, 0, target_w, target_h))


def _resolve_mask_context(content_img: Image.Image, mask_path: str):
    original = content_img.convert("RGB")
    mask = load_mask_image(mask_path, original.size)
    bbox = get_mask_bbox(mask)
    print(f"[Clothing] selected mask path: {mask_path}")
    print(f"[Clothing] selected mask bbox: {bbox}")
    if bbox is None:
        raise gr.Error("Selected garment mask is empty.")
    return original, mask, bbox


def apply_palette_recolor_transfer(
    content_img: Image.Image,
    mask_path: str,
    selected_color: str,
    style_strength: float,
) -> Image.Image:
    original, mask, bbox = _resolve_mask_context(content_img, mask_path)
    print("[Clothing] applying palette recolor as true recolor, not overlay")
    print(f"[Clothing] style strength: {style_strength}")

    style_strength = float(np.clip(style_strength, 0.0, 1.0))
    target_rgb = parse_hex_color(selected_color)

    original_roi = original.crop(bbox)
    orig_np = np.array(original_roi).astype(np.float32)
    mask_np = (np.array(mask.crop(bbox).convert("L")).astype(np.float32) / 255.0)[
        ..., None
    ]

    luminance = (
        0.2126 * orig_np[:, :, 0]
        + 0.7152 * orig_np[:, :, 1]
        + 0.0722 * orig_np[:, :, 2]
    )
    active_mask = mask_np[:, :, 0] > 0
    if not np.any(active_mask):
        raise gr.Error("Selected garment mask is empty.")

    masked_luminance = luminance[active_mask]
    p5 = float(np.percentile(masked_luminance, 5))
    p95 = float(np.percentile(masked_luminance, 95))
    print(f"[Clothing] luminance p5/p95: {p5:.4f} / {p95:.4f}")
    if p95 - p5 < 1e-6:
        luminance_norm = np.clip(luminance / 255.0, 0.0, 1.0)
    else:
        luminance_norm = np.clip((luminance - p5) / (p95 - p5), 0.0, 1.0)

    shading = 0.55 + (0.70 * luminance_norm)
    recolor_target = np.clip(
        target_rgb.reshape((1, 1, 3)) * shading[..., None],
        0,
        255,
    )
    blended = orig_np * (1.0 - style_strength) + recolor_target * style_strength
    blended = (orig_np * (1.0 - mask_np)) + (blended * mask_np)

    base = original.copy()
    base.paste(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), bbox)
    return composite_result_with_mask(original, base, mask_path)


def apply_pattern_print_transfer(
    content_img: Image.Image,
    style_img_path: str,
    mask_path: str,
    style_strength: float,
    pattern_scale: float,
) -> Image.Image:
    original, mask, bbox = _resolve_mask_context(content_img, mask_path)
    style_img = _load_style_reference(style_img_path)
    print("[Clothing] Pattern Print Transfer selected")
    print("[Clothing] applying pattern print transfer only inside selected mask")

    style_strength = float(np.clip(style_strength, 0.0, 1.0))
    pattern_scale = float(np.clip(pattern_scale, 0.3, 4.0))
    print(f"[Clothing] pattern strength: {style_strength}")
    print(f"[Clothing] pattern size / scale: {pattern_scale}")
    left, upper, right, lower = bbox
    roi_size = (right - left, lower - upper)
    base_tile_size = max(32.0, min(roi_size[0], roi_size[1]) * 0.45)
    tile_size = int(
        np.clip(
            base_tile_size * pattern_scale,
            32,
            max(roi_size[0], roi_size[1]),
        )
    )
    print(f"[Clothing] computed tile size: {tile_size}")
    pattern_img = _tile_style_to_size(style_img, roi_size, tile_size)

    original_roi = original.crop(bbox)
    orig_np = np.array(original_roi).astype(np.float32)
    pattern_np = np.array(pattern_img).astype(np.float32)
    mask_np = (np.array(mask.crop(bbox).convert("L")).astype(np.float32) / 255.0)[
        ..., None
    ]

    luminance = np.mean(orig_np, axis=2, keepdims=True) / 255.0
    luminance = 0.55 + (0.45 * luminance)
    structured_pattern = np.clip(pattern_np * luminance, 0, 255)
    blended = orig_np * (1.0 - style_strength) + structured_pattern * style_strength
    blended = (orig_np * (1.0 - mask_np)) + (blended * mask_np)

    base = original.copy()
    base.paste(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)), bbox)
    return composite_result_with_mask(original, base, mask_path)


def merge_selected_masks(
    selected_mask_names: List[str],
    mask_state: Dict[str, object],
) -> Optional[str]:
    mask_paths = (mask_state or {}).get("mask_paths") or {}
    image_path = (mask_state or {}).get("image_path")
    session_dir = (mask_state or {}).get("session_dir")
    if not selected_mask_names or not mask_paths or not image_path or not session_dir:
        return None

    source_image = Image.open(image_path).convert("RGB")
    merged = np.zeros((source_image.height, source_image.width), dtype=np.uint8)

    for mask_name in selected_mask_names:
        mask_path = mask_paths.get(mask_name)
        if not mask_path or (not os.path.isfile(mask_path)):
            continue

        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != source_image.size:
            mask_img = mask_img.resize(source_image.size, Image.NEAREST)
        merged = np.maximum(merged, (np.array(mask_img) > 0).astype(np.uint8) * 255)

    if not np.any(merged > 0):
        return None

    merged_mask_path = Path(session_dir) / "merged_selected_mask.png"
    Image.fromarray(merged).save(merged_mask_path)
    mask_state["merged_mask_path"] = str(merged_mask_path)
    return str(merged_mask_path)


def update_mask_preview(
    selected_mask_names: List[str],
    mask_state: Dict[str, object],
):
    selected_mask_names = selected_mask_names or []
    print(f"[Clothing] selected garment regions: {selected_mask_names}")

    if not mask_state or not mask_state.get("image_path"):
        return None, mask_state, None

    mask_state["selected_mask_names"] = selected_mask_names
    merged_mask_path = merge_selected_masks(selected_mask_names, mask_state)
    print(f"[Clothing] merged mask path: {merged_mask_path}")

    if merged_mask_path and os.path.isfile(merged_mask_path):
        merged_mask = np.array(Image.open(merged_mask_path).convert("L"))
        preview = build_mask_preview(mask_state["image_path"], merged_mask)
    else:
        preview = build_mask_preview(mask_state["image_path"], None)

    print("[Clothing] mask preview updated")
    return preview, mask_state, merged_mask_path


def update_clothing_mask_selection(
    selected_mask_names: List[str],
    mask_state: Dict[str, object],
):
    preview, updated_mask_state, merged_mask_path = update_mask_preview(
        selected_mask_names, mask_state
    )
    if updated_mask_state is not None:
        updated_mask_state["selected_mask_names"] = selected_mask_names or []
        updated_mask_state["merged_mask_path"] = merged_mask_path
    has_selection = bool(selected_mask_names) and bool(merged_mask_path)
    apply_hint = (
        _step_apply_hint_text()
        if has_selection
        else "Step 5 is disabled until you select at least one garment region."
    )
    return (
        preview,
        updated_mask_state,
        merged_mask_path,
        apply_hint,
    )


def _empty_palette_ui_state():
    return (
        gr.update(value=[]),
        [],
        "",
        "",
        build_color_preview_html(""),
    )


def _empty_palette_nst_ui_state():
    return (
        gr.update(choices=[], value=None),
        None,
        [],
        "",
    )


def _empty_garment_ui_state():
    return (
        gr.update(visible=False),
        gr.update(choices=[], value=[]),
        None,
        {},
        None,
    )


def _empty_clothing_result():
    return None


def _step_apply_hint_text():
    return "Step 5 is ready after you select at least one garment region in Step 2."


def _person_background_apply_hint_text():
    return "Background target uses safe background mask."


def _person_default_apply_hint_text():
    return "Please generate garment regions and select at least one region before applying style."


def _empty_palette_gallery_ui():
    palette_gallery_update, palette_empty_update = palette_placeholder_updates([])
    return (
        palette_gallery_update,
        palette_empty_update,
        [],
        "",
        "",
        build_color_preview_html(""),
    )


def _person_region_masks_reset_update(mask_state: Optional[Dict[str, object]] = None):
    choices = []
    if mask_state and mask_state.get("mask_paths"):
        choices = list(mask_state["mask_paths"].keys())
    return gr.update(choices=choices, value=[], visible=True)


def _person_region_masks_hidden_update(mask_state: Optional[Dict[str, object]] = None):
    choices = []
    if mask_state and mask_state.get("mask_paths"):
        choices = list(mask_state["mask_paths"].keys())
    return gr.update(choices=choices, value=[], visible=False)


def on_person_content_change(content_img: Optional[Image.Image]):
    # outputs:
    # 0 person_region_masks
    # 1 person_segmentation_preview
    # 2 person_selected_preview
    # 3 person_mask_state
    # 4 person_merged_mask_path
    # 5 person_palette_gallery
    # 6 person_palette_empty_state
    # 7 person_palette_colors
    # 8 person_custom_color
    # 9 person_selected_palette_color
    # 10 person_custom_color_preview
    # 11 person_result
    # 12 person_region_hint
    # 13 person_apply_hint
    (
        _group_update,
        _mask_update,
        _segmentation_preview,
        _selected_preview,
        mask_state,
        _merged_mask_path,
        _palette_gallery,
        _palette_colors,
        _custom_color,
        _selected_theme_color,
        _selected_color_preview,
        _result_image,
        _region_hint,
        _apply_hint,
    ) = prepare_clothing_tab_on_image_change(content_img)
    segmentation_image_update = gr.update(value=None, visible=True)
    selected_image_update = gr.update(value=None, visible=True)
    result_image_update = gr.update(value=None, visible=True)
    (
        palette_gallery_update,
        palette_empty_update,
        palette_colors,
        custom_color,
        selected_palette_color,
        selected_color_preview,
    ) = _empty_palette_gallery_ui()
    return (
        _person_region_masks_reset_update(),
        segmentation_image_update,
        selected_image_update,
        mask_state,
        None,
        palette_gallery_update,
        palette_empty_update,
        palette_colors,
        custom_color,
        selected_palette_color,
        selected_color_preview,
        result_image_update,
        "",
        "",
    )


def on_detect_person_regions(
    content_img: Optional[Image.Image],
    edit_target: str,
):
    # outputs:
    # 0 person_region_masks
    # 1 person_segmentation_preview
    # 2 person_selected_preview
    # 3 person_mask_state
    # 4 person_merged_mask_path
    # 5 person_palette_gallery
    # 6 person_palette_empty_state
    # 7 person_palette_colors
    # 8 person_custom_color
    # 9 person_selected_palette_color
    # 10 person_custom_color_preview
    # 11 person_result
    # 12 person_region_hint
    # 13 person_apply_hint
    (
        _group_update,
        _mask_update,
        segmentation_preview,
        _selected_preview,
        mask_state,
        _merged_mask_path,
        _result_image,
        region_hint,
        _apply_hint,
    ) = generate_person_region_masks_for_tab(content_img)
    detected_preview = None
    if mask_state:
        label_map_preview_path = mask_state.get("label_map_preview_path")
        if label_map_preview_path and os.path.isfile(label_map_preview_path):
            detected_preview = Image.open(label_map_preview_path).convert("RGB")
    if detected_preview is None:
        detected_preview = segmentation_preview
    is_background = edit_target == "Background"
    safe_background_preview = None
    if is_background and mask_state:
        safe_background_preview_path = mask_state.get("safe_background_preview_path")
        if safe_background_preview_path and os.path.isfile(
            safe_background_preview_path
        ):
            safe_background_preview = Image.open(safe_background_preview_path).convert(
                "RGB"
            )
    segmentation_image_update = gr.update(value=detected_preview, visible=True)
    selected_image_update = gr.update(
        value=safe_background_preview if is_background else None,
        visible=True,
    )
    result_image_update = gr.update(value=None, visible=True)
    (
        palette_gallery_update,
        palette_empty_update,
        palette_colors,
        custom_color,
        selected_palette_color,
        selected_color_preview,
    ) = _empty_palette_gallery_ui()
    ret = (
        gr.update(
            choices=list((mask_state or {}).get("mask_paths", {}).keys()),
            value=[],
            visible=not is_background,
        ),
        segmentation_image_update,
        selected_image_update,
        mask_state,
        (
            mask_state.get("safe_background_mask_path")
            if is_background and isinstance(mask_state, dict)
            else None
        ),
        palette_gallery_update,
        palette_empty_update,
        palette_colors,
        custom_color,
        selected_palette_color,
        selected_color_preview,
        result_image_update,
        "",
        "",
    )
    print(
        "[Detect] mask_state keys =",
        list(mask_state.keys()) if isinstance(mask_state, dict) else type(mask_state),
    )
    print(
        "[Detect] label_map_preview_path =",
        (
            mask_state.get("label_map_preview_path")
            if isinstance(mask_state, dict)
            else None
        ),
    )
    print(
        "[Detect] detected_preview type =",
        type(detected_preview).__name__ if detected_preview is not None else None,
    )
    print("[Detect] output count =", len(ret))
    print("[Detect] return types =", [type(x).__name__ for x in ret])
    return ret


def prepare_person_tab_on_image_change_ui(
    content_img: Optional[Image.Image],
):
    ret = on_person_content_change(content_img)
    log_callback_return("prepare_person_tab_on_image_change_ui", ret)
    return ret


def generate_person_region_masks_for_tab_ui(
    content_img: Optional[Image.Image],
):
    ret = on_detect_person_regions(content_img, "Clothing / Accessories")
    log_callback_return("generate_person_region_masks_for_tab_ui", ret)
    return ret


def update_person_region_selection_ui(
    selected_mask_names: List[str],
    mask_state: Dict[str, object],
):
    # outputs:
    # 0 person_selected_preview
    # 1 person_mask_state
    # 2 person_merged_mask_path
    # 3 person_apply_hint
    print("[Selection] selected_mask_names =", selected_mask_names)
    print(
        "[Selection] mask_state keys =",
        list(mask_state.keys()) if isinstance(mask_state, dict) else type(mask_state),
    )
    print(
        "[Selection] mask_paths =",
        mask_state.get("mask_paths") if isinstance(mask_state, dict) else None,
    )
    selected_mask_names = selected_mask_names or []
    preview, updated_mask_state, merged_mask_path, apply_hint = (
        update_clothing_mask_selection(
            selected_mask_names,
            mask_state,
        )
    )
    if updated_mask_state is None:
        updated_mask_state = {}
    updated_mask_state["selected_mask_names"] = selected_mask_names or []
    updated_mask_state["merged_mask_path"] = merged_mask_path
    if (
        not selected_mask_names
        or not merged_mask_path
        or not os.path.isfile(merged_mask_path)
    ):
        preview = None
        apply_hint = _person_default_apply_hint_text()
    preview_update = gr.update(value=preview, visible=True)
    selected_preview = preview
    ret = (
        preview_update,
        updated_mask_state,
        merged_mask_path,
        apply_hint,
    )
    print("[Selection] merged_mask_path =", merged_mask_path)
    print(
        "[Selection] selected_preview type =",
        type(selected_preview).__name__ if selected_preview is not None else None,
    )
    print(f"[UI] outputs = {4}")
    print(f"[UI] returns = {len(ret)}")
    print(f"[UI] return types = {[type(x).__name__ for x in ret]}")
    log_callback_return("update_person_region_selection_ui", ret)
    return ret


def update_person_target_selection_ui(
    edit_target: str,
    mask_state: Dict[str, object],
):
    # outputs:
    # 0 person_clothing_mode_group
    # 1 person_background_mode_group
    # 2 person_advanced_settings
    # 3 person_region_masks_group
    # 4 person_region_masks
    # 5 person_selected_preview
    # 6 person_merged_mask_path
    # 7 person_apply_hint
    is_background = edit_target == "Background"
    preview = None
    if is_background and mask_state:
        safe_preview_path = mask_state.get("safe_background_preview_path")
        if safe_preview_path and os.path.isfile(safe_preview_path):
            preview = Image.open(safe_preview_path).convert("RGB")
    elif mask_state:
        selected_mask_names = mask_state.get("selected_mask_names") or []
        merged_mask_path = mask_state.get("merged_mask_path")
        if (
            selected_mask_names
            and merged_mask_path
            and os.path.isfile(merged_mask_path)
        ):
            preview = build_mask_preview(
                mask_state["image_path"],
                np.array(Image.open(merged_mask_path).convert("L")),
            )

    if is_background:
        ret = (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(
                choices=list((mask_state or {}).get("mask_paths", {}).keys()),
                value=[],
                visible=False,
            ),
            gr.update(value=preview, visible=True),
            (
                mask_state.get("safe_background_mask_path")
                if isinstance(mask_state, dict)
                else None
            ),
            "",
        )
    else:
        if not (
            (mask_state or {}).get("selected_mask_names")
            and (mask_state or {}).get("merged_mask_path")
        ):
            preview = None
        ret = (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(
                choices=list((mask_state or {}).get("mask_paths", {}).keys()),
                value=(mask_state or {}).get("selected_mask_names") or [],
                visible=True,
            ),
            gr.update(value=preview, visible=True),
            (mask_state or {}).get("merged_mask_path"),
            "",
        )
    print(f"[UI] outputs = {8}")
    print(f"[UI] returns = {len(ret)}")
    print(f"[UI] return types = {[type(x).__name__ for x in ret]}")
    log_callback_return("update_person_target_selection_ui", ret)
    return ret


def extract_person_palette_ui(
    style_img_path: Optional[str],
    garment_mask_state: Dict[str, object],
):
    (
        _gallery_update,
        colors,
        custom_color,
        selected_color,
        color_preview_html,
        updated_mask_state,
    ) = extract_palette_for_session(style_img_path, garment_mask_state)
    gallery_items = build_palette_gallery_items(
        colors,
        selected_color=selected_color if colors else None,
    )
    palette_gallery_update, palette_empty_update = palette_placeholder_updates(
        gallery_items
    )
    outputs = (
        palette_gallery_update,
        palette_empty_update,
        colors,
        custom_color,
        selected_color,
        color_preview_html,
        updated_mask_state,
    )
    log_callback_return("extract_person_palette_ui", outputs)
    return outputs


def apply_person_aware_local_edit_ui(*args, **kwargs):
    (
        result_image,
        _group_update,
        _masks_update,
        _preview_update,
        mask_state,
        merged_mask_path,
        _region_hint_update,
        apply_hint_update,
    ) = apply_person_aware_local_edit_with_mode_strength(*args, **kwargs)
    outputs = (
        gr.update(value=result_image, visible=True),
        mask_state,
        merged_mask_path,
        apply_hint_update,
    )
    log_callback_return("apply_person_aware_local_edit_ui", outputs)
    return outputs


def run_nst_ui(*args, **kwargs):
    result_image, *_ = run_nst(*args, **kwargs)
    result_image_update, result_empty_update = show_image_or_empty(
        result_image,
        "Your stylized result will appear here.",
        min_height=440,
    )
    outputs = (result_image_update, result_empty_update)
    log_callback_return("run_nst_ui", outputs)
    return outputs


def run_sd_only_ui(*args, **kwargs):
    result_image, *_ = run_sd_only(*args, **kwargs)
    return gr.update(value=result_image)


def prepare_clothing_tab_on_image_change(
    content_img: Optional[Image.Image],
):
    group_update, mask_update, preview, mask_state, merged_mask_path = (
        prepare_garment_ui(content_img, "Clothing / Fashion")
    )
    (
        palette_gallery,
        palette_colors,
        custom_color,
        selected_theme_color,
        selected_color_preview,
    ) = _empty_palette_ui_state()
    region_hint = ""
    return (
        group_update,
        mask_update,
        preview,
        None,
        mask_state,
        merged_mask_path,
        palette_gallery,
        palette_colors,
        custom_color,
        selected_theme_color,
        selected_color_preview,
        _empty_clothing_result(),
        region_hint,
        "Step 5 is disabled until you finish Step 1 and Step 2.",
    )


def prepare_garment_ui(content_img: Optional[Image.Image], scene_preset: str):
    if not is_clothing_scene(scene_preset):
        return _empty_garment_ui_state()

    print("[Clothing] Clothing mode selected, show garment region UI")
    preview = content_img.convert("RGB") if content_img is not None else None
    return (
        gr.update(visible=True),
        gr.update(choices=[], value=[]),
        preview,
        {},
        None,
    )


def segment_current_image_for_ui(
    content_img: Optional[Image.Image],
    scene_preset: str,
):
    if not is_clothing_scene(scene_preset):
        return _empty_garment_ui_state()

    print("[Clothing] Clothing mode selected, show garment region UI")
    if content_img is None:
        raise gr.Error("Please upload a content image first.")

    try:
        mask_state = run_garment_segmentation(content_img)
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        print(f"[Clothing] garment segmentation failed: {exc}")
        raise gr.Error("Failed to generate garment masks for current image.") from exc

    mask_names = list(mask_state["mask_paths"].keys())
    preview_path = mask_state.get("label_map_preview_path")
    preview = None
    if preview_path and os.path.isfile(preview_path):
        preview = Image.open(preview_path).convert("RGB")
    else:
        preview = build_mask_preview(mask_state["image_path"], None)

    if not mask_names:
        gr.Warning("No garment masks found for current image.")
    else:
        gr.Info(
            "Garment masks generated. Please select garment regions and click "
            "Step 2: Apply Style to Selected Garment."
        )

    return (
        gr.update(visible=True),
        gr.update(choices=mask_names, value=[]),
        preview,
        mask_state,
        None,
    )


def extract_palette_for_session(
    style_img_path: Optional[str],
    garment_mask_state: Dict[str, object],
):
    if not style_img_path or not os.path.isfile(style_img_path):
        raise gr.Error("Please upload a Style Image first.")
    if not garment_mask_state or not garment_mask_state.get("session_dir"):
        raise gr.Error("Please click Step 1: Generate Garment Regions first.")

    style_img = _load_style_reference(style_img_path)
    colors = extract_style_palette(style_img, n_colors=10)
    preview_path = Path(garment_mask_state["session_dir"]) / "style_palette_preview.png"
    save_palette_preview(colors, str(preview_path))
    gallery_items = build_palette_gallery_items(
        colors, selected_color=colors[0] if colors else None
    )
    garment_mask_state["palette_colors"] = colors
    garment_mask_state["palette_preview_path"] = str(preview_path)

    default_color = colors[0] if colors else None
    print(f"[Clothing] default selected color: {default_color}")
    return (
        gr.update(value=gallery_items),
        colors,
        "",
        default_color,
        build_color_preview_html(""),
        garment_mask_state,
    )


def extract_palette_for_session_nst(
    style_img_path: Optional[str],
    garment_mask_state: Dict[str, object],
):
    if not style_img_path or not os.path.isfile(style_img_path):
        raise gr.Error("Please upload a Style Image first.")
    if not garment_mask_state or not garment_mask_state.get("session_dir"):
        raise gr.Error("Please click Step 1: Generate Garment Regions first.")

    style_img = _load_style_reference(style_img_path)
    colors = extract_style_palette(style_img, n_colors=10)
    preview_path = Path(garment_mask_state["session_dir"]) / "style_palette_preview.png"
    save_palette_preview(colors, str(preview_path))
    garment_mask_state["palette_colors"] = colors
    garment_mask_state["palette_preview_path"] = str(preview_path)

    preview = Image.open(preview_path).convert("RGB") if colors else None
    default_color = colors[0] if colors else None
    return (
        gr.update(choices=colors, value=default_color),
        preview,
        colors,
        "",
        garment_mask_state,
    )


def on_palette_gallery_select(evt: gr.SelectData, palette_colors: List[str]):
    if palette_colors is None or not isinstance(palette_colors, list):
        gr.Warning("Palette colors are not ready yet. Please extract palette again.")
        return gr.update(value=[]), ""
    if not palette_colors:
        return gr.update(value=[]), ""
    index = evt.index
    if isinstance(index, tuple):
        index = index[0]
    try:
        index = int(index)
    except (TypeError, ValueError):
        gr.Warning("Invalid palette selection.")
        return gr.update(value=[]), ""
    if index < 0 or index >= len(palette_colors):
        gr.Warning("Palette selection is out of range.")
        return gr.update(value=[]), ""
    selected_color = palette_colors[index]
    print(f"[Clothing] selected palette color: {selected_color}")
    gallery_items = build_palette_gallery_items(
        palette_colors,
        selected_color=selected_color,
    )
    return gr.update(value=gallery_items), selected_color


def update_custom_color_preview(custom_color: str):
    return build_color_preview_html((custom_color or "").strip())


def safe_float(value, default: float, name: str) -> float:
    if isinstance(value, gr.Progress):
        raise gr.Error(
            f"Internal binding error: {name} received Gradio Progress instead of a number."
        )
    if value in (None, ""):
        print(f"[PersonAware][Warning] {name} is empty, fallback to {default}")
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"Internal binding error: {name} should be numeric.") from exc


def safe_int(value, default: int, name: str) -> int:
    if isinstance(value, gr.Progress):
        raise gr.Error(
            f"Internal binding error: {name} received Gradio Progress instead of a number."
        )
    if value in (None, ""):
        print(f"[PersonAware][Warning] {name} is empty, fallback to {default}")
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"Internal binding error: {name} should be numeric.") from exc


def get_active_selected_color(
    color_source: str,
    selected_palette_color: Optional[str],
    custom_color: str,
    palette_colors: List[str],
):
    print(f"[PersonAware] active color source: {color_source}")
    if isinstance(selected_palette_color, list):
        print("[PersonAware][Error] selected_palette_color received list")
        raise gr.Error(
            "Internal binding error: selected_palette_color should be a string, but got list."
        )
    if palette_colors is None:
        palette_colors = []
    if not isinstance(palette_colors, list):
        print(
            f"[PersonAware][Error] palette_colors invalid type: {type(palette_colors)}"
        )
        raise gr.Error(
            "Internal binding error: palette_colors should be a list of hex colors."
        )

    if color_source == "Use extracted palette color":
        active_color = (selected_palette_color or "").strip()
        if not active_color and palette_colors:
            active_color = palette_colors[0]
        if not active_color:
            raise gr.Error("Please extract a palette first, or choose custom color.")
        print("[PersonAware] using selected palette color")
        print(f"[PersonAware] active selected color: {active_color}")
        return active_color

    active_color = (custom_color or "").strip()
    try:
        parse_hex_color(active_color)
    except Exception as exc:
        raise gr.Error("Please enter a valid custom color in #RRGGBB format.") from exc
    print("[PersonAware] using custom color")
    print(f"[PersonAware] active selected color: {active_color}")
    return active_color


def resolve_person_local_style_strength(
    person_style_mode: str,
    palette_strength: float,
    pattern_strength: float,
):
    if person_style_mode == "Pattern Print Transfer":
        return pattern_strength
    return palette_strength


def generate_person_region_masks_for_tab(
    content_img: Optional[Image.Image],
):
    group_update, mask_update, preview, mask_state, merged_mask_path = (
        segment_current_image_for_ui(content_img, "Clothing / Fashion")
    )
    region_hint = "Select one or more garment regions below."
    if not mask_state or not mask_state.get("mask_paths"):
        region_hint = "No garment masks found for current image."
    return (
        group_update,
        mask_update,
        preview,
        None,
        mask_state,
        merged_mask_path,
        _empty_clothing_result(),
        region_hint,
        "Step 5 is disabled until you select at least one garment region.",
    )


def apply_person_aware_local_edit(*args, **kwargs):
    result, group_update, masks_update, preview_update, mask_state, merged_mask_path = (
        apply_style_to_selected_garment(*args, **kwargs)
    )
    return (
        result,
        group_update,
        masks_update,
        preview_update,
        mask_state,
        merged_mask_path,
        "Style applied. Adjust controls and re-run Step 5 if needed.",
        _step_apply_hint_text(),
    )


def apply_person_aware_local_edit_with_mode_strength(
    content_img: Image.Image,
    style_img_path: Optional[str],
    prompt_preset: str,
    custom_prompt: str,
    palette_style_strength: float,
    pattern_style_strength: float,
    pattern_scale: float,
    enable_sd_refine: bool,
    sd_refine_strength: float,
    steps: int,
    guidance: float,
    scene_preset: str,
    color_tone: str,
    person_style_mode: str,
    color_source: str,
    selected_palette_color: Optional[str],
    custom_color: str,
    selected_mask_names: List[str],
    garment_mask_state: Dict[str, object],
    merged_garment_mask_path: Optional[str],
    person_edit_target: str,
    background_mode: str,
    background_prompt: str,
    use_style_palette_for_background: bool,
    palette_colors: List[str],
    background_strength: float,
    background_steps: int,
    background_guidance: float,
    background_additional_instruction: str,
    background_fit_mode: str,
    background_blend_strength: float,
    progress=gr.Progress(),
):
    print(
        f"[PersonAware] apply args | selected_palette_color type={type(selected_palette_color)} "
        f"value={selected_palette_color}"
    )
    print(
        f"[PersonAware] apply args | palette_colors type={type(palette_colors)} "
        f"value={palette_colors}"
    )
    print(
        f"[PersonAware] apply args | background_blend_strength type={type(background_blend_strength)} "
        f"value={background_blend_strength}"
    )
    print(
        f"[PersonAware] apply args | background_guidance type={type(background_guidance)} "
        f"value={background_guidance}"
    )
    print(
        f"[PersonAware] apply args | background_steps type={type(background_steps)} "
        f"value={background_steps}"
    )
    print(
        f"[PersonAware] apply args | background_strength type={type(background_strength)} "
        f"value={background_strength}"
    )
    if isinstance(selected_palette_color, list):
        raise gr.Error(
            "Internal binding error: selected_palette_color should be a string, but got list."
        )
    if palette_colors is not None and not isinstance(palette_colors, list):
        raise gr.Error(
            "Internal binding error: palette_colors should be a list of hex colors."
        )

    if person_edit_target == "Background":
        if garment_mask_state and garment_mask_state.get("safe_background_mask_path"):
            merged_garment_mask_path = garment_mask_state["safe_background_mask_path"]
        result = apply_background_repaint(
            content_img=content_img,
            style_img_path=style_img_path,
            background_mode=background_mode,
            background_prompt=background_prompt,
            use_style_palette=use_style_palette_for_background,
            color_source=color_source,
            selected_palette_color=selected_palette_color,
            custom_color=custom_color,
            palette_colors=palette_colors,
            background_strength=background_strength,
            steps=background_steps,
            guidance=background_guidance,
            background_additional_instruction=background_additional_instruction,
            background_fit_mode=background_fit_mode,
            background_blend_strength=background_blend_strength,
            garment_mask_state=garment_mask_state,
            progress=progress,
        )
        return (
            result,
            gr.update(),
            gr.update(),
            gr.update(),
            garment_mask_state,
            merged_garment_mask_path,
            "Background repaint applied.",
            "Background target uses safe background mask.",
        )

    resolved_strength = resolve_person_local_style_strength(
        person_style_mode,
        palette_style_strength,
        pattern_style_strength,
    )
    return apply_person_aware_local_edit(
        content_img,
        style_img_path,
        prompt_preset,
        custom_prompt,
        resolved_strength,
        pattern_scale,
        enable_sd_refine,
        sd_refine_strength,
        steps,
        guidance,
        scene_preset,
        color_tone,
        person_style_mode,
        color_source,
        selected_palette_color,
        custom_color,
        selected_mask_names,
        garment_mask_state,
        merged_garment_mask_path,
        progress=progress,
    )


def get_selected_garment_mask_path(
    scene_preset: str,
    selected_mask_names: List[str],
    merged_mask_path: Optional[str],
) -> Optional[str]:
    if not is_clothing_scene(scene_preset):
        return None

    print("[Clothing] Clothing mode selected, show garment region UI")
    selected_mask_names = selected_mask_names or []
    if not selected_mask_names:
        print("[Clothing] no selected garment region -> stop generation")
        raise gr.Error("Please select at least one garment region.")

    print(f"[Clothing] selected garment regions: {selected_mask_names}")
    print(f"[Clothing] merged mask path: {merged_mask_path}")
    if not merged_mask_path or (not os.path.isfile(merged_mask_path)):
        print("[Clothing] no selected garment region -> stop generation")
        raise gr.Error("Please select at least one garment region.")
    return merged_mask_path


def _unchanged_clothing_outputs(
    result_image,
    mask_state: Dict[str, object],
    merged_mask_path: Optional[str],
):
    return (
        result_image,
        gr.update(),
        gr.update(),
        gr.update(),
        mask_state,
        merged_mask_path,
    )


def _effective_clothing_strength(strength: float) -> float:
    return min(max(float(strength), 0.25), 0.35)


def apply_style_to_selected_garment(
    content_img: Image.Image,
    style_img_path: Optional[str],
    prompt_preset: str,
    custom_prompt: str,
    style_strength: float,
    pattern_scale: float,
    enable_sd_refine: bool,
    sd_refine_strength: float,
    steps: int,
    guidance: float,
    scene_preset: str,
    color_tone: str,
    clothing_style_mode: str,
    color_source: str,
    selected_theme_color: Optional[str],
    custom_color: str,
    selected_mask_names: List[str],
    garment_mask_state: Dict[str, object],
    merged_garment_mask_path: Optional[str],
    progress=gr.Progress(),
):
    if content_img is None:
        raise gr.Error("Please upload a content image first.")
    if not is_clothing_scene(scene_preset):
        raise gr.Error(
            "Clothing style apply is only available in Clothing / Fashion mode."
        )
    if not garment_mask_state or not garment_mask_state.get("session_dir"):
        raise gr.Error("Please click Step 1: Generate Garment Regions first.")
    if not garment_mask_state.get("mask_paths"):
        raise gr.Error("No garment masks found for current image.")
    if selected_mask_names and garment_mask_state:
        recomputed_mask_path = merge_selected_masks(
            selected_mask_names,
            garment_mask_state,
        )
        if recomputed_mask_path:
            merged_garment_mask_path = recomputed_mask_path
            garment_mask_state["merged_mask_path"] = recomputed_mask_path
            garment_mask_state["selected_mask_names"] = selected_mask_names

    garment_mask_path = get_selected_garment_mask_path(
        scene_preset=scene_preset,
        selected_mask_names=selected_mask_names,
        merged_mask_path=merged_garment_mask_path,
    )
    final_prompt = get_final_prompt(prompt_preset, custom_prompt)
    original = content_img.convert("RGB")
    selected_bbox = get_mask_bbox(load_mask_image(garment_mask_path, original.size))
    print(f"[Clothing] selected mask path: {garment_mask_path}")
    print(f"[Clothing] selected mask bbox: {selected_bbox}")
    print(f"[Clothing] clothing style mode: {clothing_style_mode}")
    print(f"[Clothing] style strength: {style_strength}")
    print(f"[Clothing] sd refine enabled: {enable_sd_refine}")

    progress(0.0, desc="Applying selected garment style...")
    if clothing_style_mode == "Palette Recolor Transfer":
        if (
            color_source == "Use extracted palette color"
            and not selected_theme_color
            and style_img_path
            and os.path.isfile(style_img_path)
        ):
            style_img = _load_style_reference(style_img_path)
            colors = extract_style_palette(style_img, n_colors=10)
            if colors:
                selected_theme_color = colors[0]
                garment_mask_state["palette_colors"] = colors

        selected_color = get_active_selected_color(
            color_source=color_source,
            selected_palette_color=selected_theme_color,
            custom_color=custom_color,
            palette_colors=garment_mask_state.get("palette_colors") or [],
        )

        base_result = apply_palette_recolor_transfer(
            original,
            garment_mask_path,
            selected_color,
            style_strength,
        )
    else:
        base_result = apply_pattern_print_transfer(
            original,
            style_img_path,
            garment_mask_path,
            style_strength,
            pattern_scale,
        )

    if not enable_sd_refine:
        progress(1.0, desc="DONE")
        return _unchanged_clothing_outputs(
            base_result,
            garment_mask_state,
            merged_garment_mask_path,
        )

    refiner = get_diffusion_refiner()
    effective_strength = _effective_clothing_strength(sd_refine_strength)
    print("[Clothing] applying inpaint only inside selected mask")

    def clothing_progress(p):
        progress(0.45 + (p * 0.55), desc="Refining selected garment...")

    refined = run_diffusion_task(
        "inpaint",
        lambda: refiner.refine(
            init_image=base_result,
            prompt=final_prompt,
            strength=effective_strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            keep_resolution=True,
            scene_preset="Clothing / Fashion",
            color_tone=color_tone if color_tone != "None" else None,
            style_image_path=style_img_path,
            blend_style_in_roi=False,
            garment_mask_path=garment_mask_path,
            progress_callback=clothing_progress,
        ),
    )
    result = composite_result_with_mask(original, refined, garment_mask_path)
    progress(1.0, desc="DONE")
    return _unchanged_clothing_outputs(
        result,
        garment_mask_state,
        merged_garment_mask_path,
    )


def run_nst(
    content_img: Image.Image,
    style_img_path: str,
    iterations: int,
    style_weight: float,
    content_weight: float,
    use_diffusion: bool,
    prompt_preset: str,
    custom_prompt: str,
    strength: float,
    steps: int,
    guidance: float,
    scene_preset: str,
    color_tone: str,
    selected_mask_names: Optional[List[str]] = None,
    garment_mask_state: Optional[Dict[str, object]] = None,
    merged_garment_mask_path: Optional[str] = None,
    progress=gr.Progress(),
):
    selected_mask_names = selected_mask_names or []
    garment_mask_state = garment_mask_state or {}
    final_prompt = get_final_prompt(prompt_preset, custom_prompt)
    garment_mask_path = None

    if is_clothing_scene(scene_preset):
        raise gr.Error(
            "Use 'Step 1: Generate Garment Regions' and "
            "'Step 2: Apply Style to Selected Garment' for Clothing / Fashion mode."
        )

    progress(0.0, desc="Style Transfer running...")
    nst_max_ratio = 0.9
    result_tensor = None
    result_img = None
    nst_runner = None

    def nst_progress_wrapper(p):
        progress(p * nst_max_ratio, desc="VGG Style Transfer running ...")

    try:
        print("[NST] before NST")
        if use_diffusion:
            print(
                "[NST] NST device selected: cpu because Stable Diffusion refine is enabled"
            )
            nst_runner = NeuralStyleTransfer()
            nst_runner.device = torch.device("cpu")
            if hasattr(nst_runner, "model") and nst_runner.model is not None:
                nst_runner.model = nst_runner.model.to(nst_runner.device)
        else:
            nst_runner = get_nst_model()
        result_tensor = nst_runner.run_style_transfer(
            content_img,
            style_img_path,
            num_iterations=iterations,
            style_weight=style_weight,
            content_weight=content_weight,
            progress_callback=nst_progress_wrapper,
        )
        print("[NST] after NST")
        result_img = tensor_to_image(result_tensor).convert("RGB")

        if not use_diffusion:
            progress(1.0, desc="DONE")
            return _unchanged_clothing_outputs(
                result_img,
                garment_mask_state,
                merged_garment_mask_path,
            )

        release_nst_memory(result_tensor)
        result_tensor = None
        if nst_runner is not None:
            release_nst_memory(nst_runner)
            nst_runner = None

        print("[NST] before diffusion")
        gpu_free_gb = get_gpu_free_gb()
        if gpu_free_gb is not None and gpu_free_gb < 4.5:
            raise gr.Error(
                "Not enough GPU memory for Stable Diffusion after NST. Try restarting Gradio or disable SD refine."
            )

        def diffusion_progress_wrapper(p):
            total_p = nst_max_ratio + (p * (1.0 - nst_max_ratio))
            progress(total_p, desc="Stable Diffusion refining ...")

        progress(nst_max_ratio, desc="Stable Diffusion refining...")
        refiner = get_diffusion_refiner()
        refiner.unload_all_pipes()
        refiner.clear_cuda_memory()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print("[NST] before NST+SD diffusion")
        log_cuda_memory("before NST+SD diffusion task")

        refined = run_diffusion_task(
            "img2img",
            lambda: refiner.refine(
                init_image=result_img,
                prompt=final_prompt,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                keep_resolution=True,
                scene_preset=scene_preset if scene_preset != "None" else None,
                color_tone=color_tone if color_tone != "None" else None,
                style_image_path=style_img_path,
                blend_style_in_roi=True,
                garment_mask_path=garment_mask_path,
                progress_callback=diffusion_progress_wrapper,
            ),
        )
        print("[NST] after NST+SD diffusion")
        progress(1.0, desc="DONE")
        return _unchanged_clothing_outputs(
            refined,
            garment_mask_state,
            merged_garment_mask_path,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise gr.Error(
            "GPU memory is not enough for NST + Stable Diffusion. Try fewer NST iterations, fewer SD steps, or restart Gradio."
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        if any(
            token in message
            for token in [
                "CUDA out of memory",
                "DefaultCPUAllocator",
                "not enough memory",
            ]
        ):
            raise gr.Error(
                "GPU memory is not enough for NST + Stable Diffusion. Try fewer NST iterations, fewer SD steps, or restart Gradio."
            ) from exc
        raise
    finally:
        release_nst_memory(result_tensor)
        release_nst_memory(nst_runner)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print("[NST] after NST+SD cleanup")
        log_cuda_memory("after NST+SD cleanup")


def run_sd_only(
    content_img: Image.Image,
    prompt_preset: str,
    custom_prompt: str,
    strength: float,
    steps: int,
    guidance: float,
    scene_preset: str,
    color_tone: str,
    style_img_path_sd: Optional[str],
    selected_mask_names: Optional[List[str]] = None,
    garment_mask_state: Optional[Dict[str, object]] = None,
    merged_garment_mask_path: Optional[str] = None,
    progress=gr.Progress(),
):
    selected_mask_names = selected_mask_names or []
    garment_mask_state = garment_mask_state or {}
    final_prompt = get_final_prompt(prompt_preset, custom_prompt)
    if is_clothing_scene(scene_preset):
        raise gr.Error(
            "Stable Diffusion Img2Img does not support Clothing / Fashion in this tab. "
            "Use Person-aware Local Edit for clothing/background target editing."
        )

    garment_mask_path = None

    progress(0.0, desc="Model Initializing...")

    def sd_callback(p):
        if p < 0.99:
            desc = "Sampling ..."
        elif p < 1.0:
            desc = "VAE Decoding & Post Processing..."
        else:
            desc = "DONE"
        progress(p, desc=desc)

    refiner = get_diffusion_refiner()
    refined = run_diffusion_task(
        "img2img",
        lambda: refiner.refine(
            init_image=content_img,
            prompt=final_prompt,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            keep_resolution=True,
            scene_preset=scene_preset if scene_preset != "None" else None,
            color_tone=color_tone if color_tone != "None" else None,
            style_image_path=style_img_path_sd,
            blend_style_in_roi=True,
            garment_mask_path=garment_mask_path,
            progress_callback=sd_callback,
        ),
    )
    return _unchanged_clothing_outputs(
        refined,
        garment_mask_state,
        merged_garment_mask_path,
    )


PALETTE_GALLERY_CSS = """
body, .gradio-container {
  background: #f6f7f9 !important;
  color: #111827 !important;
}

.gradio-container {
  --background-fill-primary: #f6f7f9 !important;
  --background-fill-secondary: #ffffff !important;
  --block-background-fill: #ffffff !important;
  --body-background-fill: #f6f7f9 !important;
  --block-border-color: #e5e7eb !important;
  --input-background-fill: #ffffff !important;
  max-width: 1480px !important;
  width: 96vw !important;
  margin: 0 auto !important;
  padding: 20px 28px 48px !important;
}

.app-shell,
.top-tabs,
.top-tabs > div,
.top-tabs [role="tabpanel"],
.top-tabs .tabitem,
.top-tabs .tabs,
.top-tabs .tab-nav {
  background: transparent !important;
}

.top-tabs {
  margin-bottom: 24px !important;
}

.top-tabs button,
.top-tabs [role="tab"] {
  font-size: 15.5px !important;
  font-weight: 650 !important;
}

.mn-section {
  background: #ffffff !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 30px 40px !important;
  margin: 0 0 30px 0 !important;
}

.mn-section.mn-section-compact {
  padding-top: 18px !important;
  padding-bottom: 16px !important;
  margin-bottom: 12px !important;
}

.mn-controls,
.mn-controls > .block,
.mn-controls > .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.mn-section-title {
  font-size: 22px !important;
  font-weight: 800 !important;
  line-height: 1.25 !important;
  color: #111827 !important;
  margin: 0 0 16px 0 !important;
  padding-left: 12px !important;
  border-left: 4px solid #f97316 !important;
}

.mn-section-compact .mn-section-title {
  margin-bottom: 14px !important;
}

.mn-subtitle {
  font-size: 16px !important;
  font-weight: 700 !important;
  line-height: 1.3 !important;
  color: #1f2937 !important;
  margin: 0 0 8px 0 !important;
}

.gradio-container label,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container button {
  font-size: 14.5px !important;
}

.step-help {
  font-size: 13px !important;
  color: #6b7280 !important;
  margin: 4px 0 0 !important;
}

.helper-text,
.helper-text p,
.upload-helper {
  display: none !important;
}

.mn-two-col {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  gap: 28px !important;
  align-items: start !important;
}

.mn-three-col {
  display: grid !important;
  grid-template-columns: 1fr 320px 1fr !important;
  gap: 30px !important;
  align-items: start !important;
}

.mn-controls {
  display: flex !important;
  flex-direction: column !important;
  gap: 18px !important;
  min-width: 0 !important;
}

.mn-section > .form,
.mn-section > .wrap,
.mn-inner-card > .form,
.mn-inner-card > .wrap {
  gap: 18px !important;
}

.vgg-layout,
.vgg-workflow {
  display: grid !important;
  grid-template-columns: 1fr 1fr !important;
  gap: 32px !important;
  align-items: start !important;
}

.vgg-left-column,
.vgg-right-column {
  display: flex !important;
  flex-direction: column !important;
  gap: 14px !important;
}

.vgg-result-area {
  margin-top: 8px !important;
}

.mn-inner-card {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 14px !important;
  padding: 18px 20px !important;
  box-shadow: none !important;
}

.sd-settings-grid .mn-inner-card,
.sd-settings-grid .sd-settings-card {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.sd-settings-grid .mn-inner-card > .form,
.sd-settings-grid .mn-inner-card > .wrap,
.sd-settings-grid .mn-inner-card > .block {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.nst-controls-area {
  display: flex !important;
  flex-direction: column !important;
  gap: 8px !important;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin: 0 !important;
}

.nst-controls-area .mn-subtitle {
  margin-bottom: 4px !important;
}

#nst-param-row,
.nst-param-row {
  display: flex !important;
  flex-direction: row !important;
  gap: 16px !important;
  align-items: flex-end !important;
  margin-top: 6px !important;
  max-width: 560px !important;
}

#nst-param-row > div,
.nst-param-row > div {
  min-width: 130px !important;
}

#nst-param-row .form,
#nst-param-row .block,
#nst-param-row .wrap,
.nst-param-row .form,
.nst-param-row .block,
.nst-param-row .wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

#nst-param-row input,
.nst-param-row input {
  width: 100% !important;
  height: 42px !important;
}

#nst-param-row label,
#nst-param-row .label,
.nst-param-row label,
.nst-param-row .label {
  white-space: normal !important;
  line-height: 1.25 !important;
  font-size: 14.5px !important;
}

.mn-style-grid {
  display: grid !important;
  grid-template-columns: 1fr 1fr !important;
  gap: 36px !important;
  align-items: start !important;
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 14px !important;
  padding: 20px !important;
}

.mn-palette-col,
.mn-control-col {
  background: #ffffff !important;
  border: none !important;
  padding: 0 !important;
}

.mn-control-col {
  display: flex !important;
  flex-direction: column !important;
  gap: 20px !important;
}

.color-control-column {
  display: flex !important;
  flex-direction: column !important;
  gap: 18px !important;
  background: #ffffff !important;
}

.pattern-controls,
.background-controls {
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.mn-image,
.mn-result {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 12px !important;
  box-shadow: none !important;
  padding: 0 !important;
  overflow: hidden !important;
}

.mn-image {
  min-height: 230px !important;
}

.mn-result {
  min-height: 460px !important;
}

.mn-image > .block,
.mn-image > .wrap,
.mn-image .image-container,
.mn-result > .block,
.mn-result > .wrap,
.mn-result .image-container {
  background: #ffffff !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.mn-image img,
.mn-result img {
  border-radius: 12px !important;
}

.mn-primary-btn button {
  width: 100% !important;
  max-width: none !important;
  height: 44px !important;
  border: none !important;
  border-radius: 10px !important;
  background: #f97316 !important;
  color: #ffffff !important;
  font-weight: 750 !important;
  box-shadow: none !important;
  margin: 0 !important;
  font-size: 15px !important;
}

.mn-primary-btn button:hover {
  background: #ea580c !important;
}

.mn-detect-btn button {
  max-width: 280px !important;
  margin: 0 auto !important;
  display: block !important;
}

.extract-palette-btn {
  margin-top: 14px !important;
}

.extract-palette-btn button {
  width: 100% !important;
  height: 44px !important;
  border-radius: 12px !important;
}

#person-palette-gallery,
.mn-palette-gallery {
  background: #ffffff !important;
  border: 1px solid #dbe3ee !important;
  border-radius: 14px !important;
  padding: 10px !important;
  height: 260px !important;
  min-height: 260px !important;
  max-height: 260px !important;
  overflow: hidden !important;
}

#person-palette-gallery .wrap,
#person-palette-gallery .grid-wrap,
#person-palette-gallery .grid-container,
.mn-palette-gallery .wrap,
.mn-palette-gallery .grid-wrap,
.mn-palette-gallery .grid-container {
  overflow: hidden !important;
}

#person-palette-gallery img,
.mn-palette-gallery img {
  object-fit: cover !important;
  border-radius: 10px !important;
}

.palette-empty-state .empty-state-panel,
.compact-empty-state .empty-state-panel,
.result-empty-state .empty-state-panel {
  background: #f8fafc !important;
  border: 1px dashed #cbd5e1 !important;
  border-radius: 12px !important;
  color: #64748b !important;
}

.compact-empty-state .empty-state-panel {
  min-height: 240px !important;
  padding: 0 !important;
}

.palette-empty-state,
.compact-empty-state {
  display: none !important;
}

.mn-radio-row,
#person-edit-target-radio,
#person-color-source-radio {
  width: 100% !important;
}

.mn-radio-row .wrap,
.mn-radio-row [role="radiogroup"],
#person-edit-target-radio .wrap,
#person-edit-target-radio [role="radiogroup"],
#person-color-source-radio .wrap,
#person-color-source-radio [role="radiogroup"] {
  display: flex !important;
  flex-direction: row !important;
  gap: 12px !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
}

.mn-radio-row label,
#person-edit-target-radio label,
#person-color-source-radio label {
  min-width: 0 !important;
  flex: 1 1 0 !important;
  height: 42px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  border-radius: 10px !important;
  border: 1px solid #d7dde7 !important;
  background: #ffffff !important;
  padding: 0 14px !important;
  margin: 0 !important;
}

.mn-radio-row label:has(input:checked),
#person-edit-target-radio label:has(input:checked),
#person-color-source-radio label:has(input:checked) {
  border-color: #f97316 !important;
  background: #fff7ed !important;
  font-weight: 700 !important;
}

.mn-checkbox-grid [role="group"],
.mn-checkbox-grid .wrap {
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  gap: 10px !important;
  overflow: visible !important;
}

.mn-checkbox-grid label {
  height: 40px !important;
  border-radius: 10px !important;
  border: 1px solid #d7dde7 !important;
  background: #ffffff !important;
  padding: 0 10px !important;
  justify-content: center !important;
  margin: 0 !important;
}

.mn-checkbox-grid label:has(input:checked) {
  border-color: #f97316 !important;
  background: #fff7ed !important;
  font-weight: 700 !important;
}

.mn-checkbox-grid input[type="checkbox"],
.mn-radio-row input[type="radio"],
#person-edit-target-radio input[type="radio"],
#person-color-source-radio input[type="radio"] {
  accent-color: #f97316 !important;
}

.mn-custom-color-row {
  display: grid !important;
  grid-template-columns: 1fr 64px !important;
  gap: 14px !important;
  align-items: center !important;
  margin-top: 0 !important;
}

.mn-custom-color-row > .form,
.mn-custom-color-row > .wrap {
  min-width: 0 !important;
}

.mn-custom-color-row input,
.mn-custom-color-row textarea {
  height: 44px !important;
}

.mn-color-preview {
  width: 56px !important;
  height: 56px !important;
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
  margin: 0 !important;
  padding: 0 !important;
}

.mn-color-preview .custom-color-preview {
  width: 56px !important;
  height: 56px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

.mn-color-preview .custom-color-preview-swatch {
  width: 56px !important;
  height: 56px !important;
  display: block !important;
  border-radius: 12px !important;
  border: 1px solid #d1d5db !important;
}

.sd-inference-params,
.inference-params {
  display: flex !important;
  flex-direction: column !important;
  gap: 22px !important;
}

.sd-inference-params .inference-params-row,
.inference-params .inference-params-row {
  gap: 24px !important;
}

.gradio-container .block,
.gradio-container .wrap,
.gradio-container .form {
  box-shadow: none !important;
}

.gradio-container [style*="display: none"],
.gradio-container [hidden] {
  display: none !important;
}

footer,
.gradio-container footer,
.gradio-container .footer,
[class*="built-with-gradio"],
[data-testid="footer"] {
  display: none !important;
}

@media (max-width: 960px) {
  .mn-two-col,
  .mn-three-col {
    grid-template-columns: 1fr !important;
  }

  .mn-primary-btn button {
    max-width: none !important;
  }

  .mn-radio-row .wrap,
  .mn-radio-row [role="radiogroup"],
  #person-edit-target-radio .wrap,
  #person-edit-target-radio [role="radiogroup"],
  #person-color-source-radio .wrap,
  #person-color-source-radio [role="radiogroup"] {
    flex-wrap: wrap !important;
  }
}
"""


with gr.Blocks(title="AI Image Style Transfer Tool", css=PALETTE_GALLERY_CSS) as demo:
    nst_pattern_scale = gr.State(1.0)

    with gr.Tabs(elem_classes=["top-tabs", "app-shell"]):
        with gr.Tab("VGG Style Transfer (NST)"):
            with gr.Column(elem_classes=["mn-section"]) as nst_sd_settings_group:
                gr.HTML(
                    '<div class="mn-section-title">Stable Diffusion Model Settings</div>'
                )
                with gr.Row(elem_classes=["mn-two-col", "sd-settings-grid"]):
                    with gr.Column(scale=1, elem_classes=["mn-inner-card"]):
                        gr.HTML('<div class="mn-subtitle">Prompt Settings</div>')
                        nst_prompt_preset = gr.Dropdown(
                            choices=PRESET_PROMPTS,
                            value=PRESET_PROMPTS[1],
                            label="General Prompt Presets / Custom",
                            allow_custom_value=False,
                        )
                        nst_prompt = gr.Textbox(
                            placeholder="Enter or modify your prompt here...",
                            label="Custom Prompts",
                            interactive=False,
                            visible=False,
                        )
                        nst_prompt_preset.change(
                            fn=toggle_custom_prompt,
                            inputs=[nst_prompt_preset],
                            outputs=[nst_prompt],
                        )
                        with gr.Row():
                            nst_scene_preset = gr.Dropdown(
                                choices=NST_SCENE_PRESETS,
                                value="None",
                                label="Scene Preset",
                            )
                            nst_color_tone = gr.Dropdown(
                                choices=COLOR_TONE_PRESETS,
                                value="None",
                                label="Color Tone Preset",
                            )
                    with gr.Column(scale=1, elem_classes=["mn-inner-card", "sd-inference-params"]):
                        gr.HTML('<div class="mn-subtitle">Inference Parameters</div>')
                        nst_strength = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            step=0.05,
                            value=0.35,
                            label="Strength",
                        )
                        with gr.Row(elem_classes=["inference-params-row"]):
                            nst_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                step=5,
                                value=10,
                                label="Sampling Steps",
                            )
                            nst_guidance = gr.Slider(
                                minimum=5.0,
                                maximum=15.0,
                                step=0.5,
                                value=7.0,
                                label="Guidance Scale",
                            )

            with gr.Column(elem_classes=["mn-section"]):
                gr.HTML('<div class="mn-section-title">VGG Style Transfer</div>')
                with gr.Row(elem_classes=["mn-two-col", "vgg-layout"]):
                    with gr.Column(scale=9, elem_classes=["mn-controls", "vgg-left-column"]):
                        gr.HTML('<div class="mn-subtitle">Content Image</div>')
                        content = gr.Image(
                            type="pil",
                            show_label=False,
                            container=False,
                            height=250,
                            elem_classes=["mn-image"],
                        )
                        gr.HTML('<div class="mn-subtitle">Style Image</div>')
                        style = gr.Image(
                            type="filepath",
                            show_label=False,
                            container=False,
                            height=250,
                            elem_classes=["mn-image"],
                        )
                    with gr.Column(scale=11, elem_classes=["mn-controls", "vgg-right-column"]):
                        with gr.Column(elem_classes=["mn-inner-card", "nst-controls-area"]):
                            gr.HTML('<div class="mn-subtitle">NST Controls</div>')
                            with gr.Row(elem_id="nst-param-row", elem_classes=["nst-param-row"]):
                                with gr.Column(scale=1, min_width=140):
                                    iterations = gr.Number(
                                        value=50, label="Iterations", min_width=120
                                    )
                                with gr.Column(scale=1, min_width=140):
                                    style_w = gr.Number(
                                        value=1e6, label="Style Weight", min_width=120
                                    )
                                with gr.Column(scale=1, min_width=140):
                                    content_w = gr.Number(
                                        value=2, label="Content Weight", min_width=120
                                    )
                            use_diff = gr.Checkbox(
                                value=True,
                                label="Using Stable Diffusion to Refine the Result (Recommended)",
                            )
                        btn = gr.Button(
                            "Run Style Transfer & Refinement",
                            variant="primary",
                            elem_classes=["mn-primary-btn"],
                        )
                        with gr.Column(elem_classes=["vgg-result-area"]):
                            gr.HTML('<div class="mn-subtitle">Result</div>')
                            out = gr.Image(
                                show_label=False,
                                container=False,
                                interactive=False,
                                height=420,
                                visible=False,
                                elem_classes=["mn-result"],
                            )
                            out_empty = gr.HTML(
                                value=build_empty_state_html(
                                    "No result yet. Upload images and click Run to generate the stylized output.",
                                    min_height=420,
                                ),
                                elem_classes=["result-empty-state"],
                            )

            use_diff.change(
                lambda x: gr.update(visible=x),
                inputs=[use_diff],
                outputs=[nst_sd_settings_group],
                show_progress=False,
            )
            btn.click(
                run_nst_ui,
                inputs=[
                    content,
                    style,
                    iterations,
                    style_w,
                    content_w,
                    use_diff,
                    nst_prompt_preset,
                    nst_prompt,
                    nst_strength,
                    nst_steps,
                    nst_guidance,
                    nst_scene_preset,
                    nst_color_tone,
                ],
                outputs=[
                    out,
                    out_empty,
                ],
            )

        with gr.Tab("Person-aware Local Edit"):
            person_mask_state = gr.State({})
            person_merged_mask_path = gr.State(None)
            person_palette_colors = gr.State([])
            person_selected_palette_color = gr.State("")
            person_scene_state = gr.State("Clothing / Fashion")
            person_prompt_preset_state = gr.State(PRESET_PROMPTS[0])
            person_color_tone_state = gr.State("None")

            with gr.Column(elem_classes=["mn-section"]):
                gr.HTML('<div class="mn-section-title">Step 0. Upload Images</div>')
                with gr.Row(elem_classes=["mn-two-col"]):
                    with gr.Column(scale=1):
                        gr.HTML('<div class="mn-subtitle">Content Image</div>')
                        person_content = gr.Image(
                            type="pil",
                            show_label=False,
                            container=False,
                            height=240,
                            elem_classes=["mn-image"],
                        )
                    with gr.Column(scale=1):
                        gr.HTML('<div class="mn-subtitle">Style Image</div>')
                        person_style = gr.Image(
                            type="filepath",
                            show_label=False,
                            container=False,
                            height=240,
                            elem_classes=["mn-image"],
                        )

            with gr.Column(elem_classes=["mn-section"]) as person_group:
                gr.HTML(
                    '<div class="mn-section-title">Step 1-2. Detect and Select Target</div>'
                )
                person_region_hint = gr.Markdown(
                    "", visible=False, elem_classes=["step-help compact-help"]
                )
                with gr.Row(elem_classes=["mn-three-col"]):
                    with gr.Column(
                        scale=4, min_width=320, elem_classes=["mn-controls"]
                    ):
                        gr.HTML('<div class="mn-subtitle">Detected Regions</div>')
                        person_segmentation_preview = gr.Image(
                            type="pil",
                            show_label=False,
                            container=False,
                            interactive=False,
                            height=320,
                            visible=True,
                            value=None,
                            elem_classes=["mn-image"],
                        )
                    with gr.Column(
                        scale=3,
                        min_width=280,
                        elem_classes=["mn-controls"],
                    ):
                        person_generate_btn = gr.Button(
                            "Detect Person Regions",
                            variant="primary",
                            elem_classes=["mn-primary-btn", "mn-detect-btn"],
                        )
                        with gr.Group(
                            visible=True,
                            elem_classes=["mn-controls", "mn-radio-row"],
                        ):
                            person_edit_target = gr.Radio(
                                choices=PERSON_EDIT_TARGETS,
                                value="Clothing / Accessories",
                                label="Edit Target",
                                elem_id="person-edit-target-radio",
                                elem_classes=["mn-radio-row"],
                            )
                        with gr.Group(
                            visible=True,
                            elem_classes=["mn-controls"],
                        ) as person_region_masks_group:
                            person_region_masks = gr.CheckboxGroup(
                                choices=[],
                                label="Available Regions",
                                elem_classes=["mn-checkbox-grid"],
                            )
                    with gr.Column(
                        scale=4, min_width=320, elem_classes=["mn-controls"]
                    ):
                        gr.HTML('<div class="mn-subtitle">Selected Target</div>')
                        person_selected_preview = gr.Image(
                            type="pil",
                            show_label=False,
                            container=False,
                            interactive=False,
                            height=320,
                            visible=True,
                            value=None,
                            elem_classes=["mn-image"],
                        )

            with gr.Column(
                elem_classes=["mn-section", "mn-section-compact"], visible=True
            ) as person_step3_group:
                gr.HTML('<div class="mn-section-title">Step 3. Choose Edit Mode</div>')
                with gr.Group(
                    visible=True, elem_classes=["inline-panel"]
                ) as person_clothing_mode_group:
                    person_style_mode = gr.Dropdown(
                        choices=CLOTHING_STYLE_MODES,
                        value="Palette Recolor Transfer",
                        label="Person-aware Style Mode",
                    )
                with gr.Group(
                    visible=False, elem_classes=["inline-panel"]
                ) as person_background_mode_group:
                    person_background_mode = gr.Dropdown(
                        choices=BACKGROUND_MODES,
                        value="Solid Color Background",
                        label="Background Mode",
                    )

            with gr.Column(
                elem_classes=["mn-section"], visible=True
            ) as person_step4_group:
                gr.HTML('<div class="mn-section-title">Step 4. Style Controls</div>')

                with gr.Group(
                    visible=True, elem_classes=["inline-panel"]
                ) as person_palette_controls_group:
                    with gr.Row(elem_classes=["mn-style-grid", "mn-inner-card"]):
                        with gr.Column(scale=1, elem_classes=["mn-controls", "mn-palette-col"]):
                            gr.HTML(
                                '<div class="mn-subtitle">Extracted Palette / Suggested Colors</div>'
                            )
                            person_palette_gallery = gr.Gallery(
                                label="",
                                columns=5,
                                rows=2,
                                height=260,
                                allow_preview=False,
                                show_label=False,
                                visible=False,
                                elem_id="person-palette-gallery",
                                elem_classes=["mn-palette-gallery"],
                            )
                            person_palette_empty_state = gr.HTML(
                                value="",
                                visible=False,
                                elem_classes=["palette-empty-state"],
                            )
                            person_palette_extract_btn = gr.Button(
                                "Extract Palette",
                                variant="primary",
                                elem_classes=["mn-primary-btn"],
                            )
                        with gr.Column(scale=1, elem_classes=["mn-control-col", "color-control-column"]):
                            person_color_source = gr.Radio(
                                choices=COLOR_SOURCE_OPTIONS,
                                value="Use extracted palette color",
                                label="Color Source",
                                elem_id="person-color-source-radio",
                                elem_classes=["mn-radio-row"],
                            )
                            with gr.Row(elem_classes=["mn-custom-color-row"]):
                                person_custom_color = gr.Textbox(
                                    value="",
                                    label="Custom Color",
                                    placeholder="#RRGGBB",
                                    scale=4,
                                )
                                person_custom_color_preview = gr.HTML(
                                    value=build_color_preview_html(""),
                                    scale=1,
                                    elem_classes=["mn-color-preview"],
                                )
                            person_palette_strength_group = gr.Group(
                                visible=True,
                                elem_classes=["inline-panel"],
                            )
                            with person_palette_strength_group:
                                person_palette_style_strength = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    step=0.05,
                                    value=0.75,
                                    label="Style Strength",
                                    interactive=True,
                                )

                with gr.Group(
                    elem_classes=["inline-panel", "pattern-controls"],
                    visible=False,
                ) as person_pattern_controls_group:
                    person_pattern_style_strength = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        step=0.05,
                        value=0.35,
                        label="Pattern Strength / Style Strength",
                        interactive=True,
                    )
                    person_pattern_scale = gr.Slider(
                        minimum=0.3,
                        maximum=4.0,
                        step=0.05,
                        value=1.0,
                        label="Pattern Size",
                        info="Small pattern / dense repeat  <->  Large pattern / sparse repeat",
                        interactive=True,
                    )

                with gr.Group(
                    elem_classes=["inline-panel"],
                    visible=False,
                ) as person_background_solid_controls_group:
                    gr.HTML("", visible=False)

                with gr.Group(
                    elem_classes=["inline-panel", "background-controls"],
                    visible=False,
                ) as person_background_style_image_panel:
                    person_background_fit_mode = gr.Dropdown(
                        choices=BACKGROUND_FIT_MODES,
                        value="Cover",
                        label="Fit Mode",
                    )
                    person_background_blend_strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        label="Blend Strength",
                    )

                with gr.Group(
                    elem_classes=["inline-panel", "background-controls"],
                    visible=False,
                ) as person_background_style_diffused_panel:
                    person_background_prompt = gr.Textbox(
                        value="replace the background with a soft artistic scene inspired by the style image",
                        label="Background Prompt",
                    )
                    person_background_use_style_palette = gr.Checkbox(
                        value=True,
                        label="Use Style Palette for Background",
                    )
                    person_background_strength = gr.Slider(
                        minimum=0.2,
                        maximum=0.7,
                        step=0.05,
                        value=0.45,
                        label="Background Img2Img Strength",
                        interactive=True,
                    )
                    with gr.Row():
                        person_background_steps = gr.Slider(
                            minimum=10,
                            maximum=40,
                            step=2,
                            value=20,
                            label="Sampling Steps",
                        )
                        person_background_guidance = gr.Slider(
                            minimum=4.0,
                            maximum=12.0,
                            step=0.5,
                            value=7.0,
                            label="Guidance Scale",
                        )
                    person_background_additional_instruction = gr.Textbox(
                        placeholder="Optional additional instruction or negative prompt for background...",
                        label="Optional Additional Instruction / Negative Prompt",
                        value="",
                    )

            with gr.Accordion(
                "Advanced Diffusion Settings",
                open=False,
                elem_classes=["advanced-panel"],
            ) as person_advanced_settings:
                person_additional_instruction = gr.Textbox(
                    placeholder="Optional additional instruction for local garment refine...",
                    label="Additional Instruction",
                    value="",
                )
                person_sd_refine = gr.Checkbox(
                    value=False,
                    label="Enable Stable Diffusion Refine",
                )
                person_sd_strength = gr.Slider(
                    minimum=0.25,
                    maximum=0.35,
                    step=0.05,
                    value=0.30,
                    label="SD Refine Strength",
                )
                with gr.Row():
                    person_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=45,
                        label="Sampling Steps",
                    )
                    person_guidance = gr.Slider(
                        minimum=5.0,
                        maximum=15.0,
                        step=0.5,
                        value=7.5,
                        label="Guidance Scale",
                    )
                person_strength = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    step=0.05,
                    value=0.55,
                    label="Strength",
                )

            with gr.Column(elem_classes=["mn-section"]):
                gr.HTML('<div class="mn-section-title">Step 5. Apply</div>')
                person_apply_hint = gr.Markdown(
                    "",
                    visible=False,
                    elem_classes=["step-help"],
                )
                person_apply_btn = gr.Button(
                    "Apply Local Edit",
                    variant="primary",
                    elem_classes=["mn-primary-btn"],
                )
                person_result = gr.Image(
                    show_label=False,
                    container=False,
                    interactive=False,
                    height=460,
                    visible=True,
                    value=None,
                    elem_classes=["mn-result"],
                )

            PERSON_AWARE_APPLY_INPUTS = [
                person_content,
                person_style,
                person_prompt_preset_state,
                person_additional_instruction,
                person_palette_style_strength,
                person_pattern_style_strength,
                person_pattern_scale,
                person_sd_refine,
                person_sd_strength,
                person_steps,
                person_guidance,
                person_scene_state,
                person_color_tone_state,
                person_style_mode,
                person_color_source,
                person_selected_palette_color,
                person_custom_color,
                person_region_masks,
                person_mask_state,
                person_merged_mask_path,
                person_edit_target,
                person_background_mode,
                person_background_prompt,
                person_background_use_style_palette,
                person_palette_colors,
                person_background_strength,
                person_background_steps,
                person_background_guidance,
                person_background_additional_instruction,
                person_background_fit_mode,
                person_background_blend_strength,
            ]

            person_content.change(
                on_person_content_change,
                inputs=[person_content],
                outputs=[
                    person_region_masks,
                    person_segmentation_preview,
                    person_selected_preview,
                    person_mask_state,
                    person_merged_mask_path,
                    person_palette_gallery,
                    person_palette_empty_state,
                    person_palette_colors,
                    person_custom_color,
                    person_selected_palette_color,
                    person_custom_color_preview,
                    person_result,
                    person_region_hint,
                    person_apply_hint,
                ],
            )
            person_generate_btn.click(
                on_detect_person_regions,
                inputs=[person_content, person_edit_target],
                outputs=[
                    person_region_masks,
                    person_segmentation_preview,
                    person_selected_preview,
                    person_mask_state,
                    person_merged_mask_path,
                    person_palette_gallery,
                    person_palette_empty_state,
                    person_palette_colors,
                    person_custom_color,
                    person_selected_palette_color,
                    person_custom_color_preview,
                    person_result,
                    person_region_hint,
                    person_apply_hint,
                ],
            )
            person_region_masks.input(
                update_person_region_selection_ui,
                inputs=[person_region_masks, person_mask_state],
                outputs=[
                    person_selected_preview,
                    person_mask_state,
                    person_merged_mask_path,
                    person_apply_hint,
                ],
            )
            person_edit_target.change(
                update_person_target_selection_ui,
                inputs=[person_edit_target, person_mask_state],
                outputs=[
                    person_clothing_mode_group,
                    person_background_mode_group,
                    person_advanced_settings,
                    person_region_masks_group,
                    person_region_masks,
                    person_selected_preview,
                    person_merged_mask_path,
                    person_apply_hint,
                ],
            )
            person_edit_target.change(
                update_person_edit_mode_ui,
                inputs=[
                    person_edit_target,
                    person_style_mode,
                    person_background_mode,
                ],
                outputs=[
                    person_step4_group,
                    person_palette_controls_group,
                    person_pattern_controls_group,
                    person_background_solid_controls_group,
                    person_background_style_image_panel,
                    person_background_style_diffused_panel,
                    person_palette_strength_group,
                ],
                js=RELAYOUT_JS,
            )
            person_palette_extract_btn.click(
                extract_person_palette_ui,
                inputs=[person_style, person_mask_state],
                outputs=[
                    person_palette_gallery,
                    person_palette_empty_state,
                    person_palette_colors,
                    person_custom_color,
                    person_selected_palette_color,
                    person_custom_color_preview,
                    person_mask_state,
                ],
            )
            person_palette_gallery.select(
                on_palette_gallery_select,
                inputs=[person_palette_colors],
                outputs=[
                    person_palette_gallery,
                    person_selected_palette_color,
                ],
            )
            person_custom_color.change(
                update_custom_color_preview,
                inputs=[person_custom_color],
                outputs=[person_custom_color_preview],
            )
            person_background_mode.change(
                update_person_edit_mode_ui,
                inputs=[
                    person_edit_target,
                    person_style_mode,
                    person_background_mode,
                ],
                outputs=[
                    person_step4_group,
                    person_palette_controls_group,
                    person_pattern_controls_group,
                    person_background_solid_controls_group,
                    person_background_style_image_panel,
                    person_background_style_diffused_panel,
                    person_palette_strength_group,
                ],
                js=RELAYOUT_JS,
            )
            person_style_mode.change(
                update_person_edit_mode_ui,
                inputs=[
                    person_edit_target,
                    person_style_mode,
                    person_background_mode,
                ],
                outputs=[
                    person_step4_group,
                    person_palette_controls_group,
                    person_pattern_controls_group,
                    person_background_solid_controls_group,
                    person_background_style_image_panel,
                    person_background_style_diffused_panel,
                    person_palette_strength_group,
                ],
                js=RELAYOUT_JS,
            )
            person_apply_btn.click(
                apply_person_aware_local_edit_ui,
                inputs=PERSON_AWARE_APPLY_INPUTS,
                outputs=[
                    person_result,
                    person_mask_state,
                    person_merged_mask_path,
                    person_apply_hint,
                ],
            )

        with gr.Tab("Stable Diffusion Img2Img"):
            with gr.Column(elem_classes=["mn-section"]):
                gr.HTML(
                    '<div class="mn-section-title">Stable Diffusion Model Settings</div>'
                )
                with gr.Row(elem_classes=["mn-two-col", "sd-settings-grid"]):
                    with gr.Column(scale=1, elem_classes=["mn-inner-card"]):
                        gr.HTML('<div class="mn-subtitle">Prompt Settings</div>')
                        sd_prompt_preset = gr.Dropdown(
                            choices=PRESET_PROMPTS,
                            value=PRESET_PROMPTS[1],
                            label="General Prompt Presets / Custom",
                            allow_custom_value=False,
                        )
                        sd_prompt = gr.Textbox(
                            placeholder="Enter or modify your prompt here...",
                            label="Custom Prompts",
                            interactive=False,
                            visible=False,
                        )
                        sd_prompt_preset.change(
                            fn=toggle_custom_prompt,
                            inputs=[sd_prompt_preset],
                            outputs=[sd_prompt],
                        )
                        with gr.Row():
                            sd_scene_preset = gr.Dropdown(
                                choices=SCENE_PRESETS,
                                value="None",
                                label="Scene Preset",
                            )
                            sd_color_tone = gr.Dropdown(
                                choices=COLOR_TONE_PRESETS,
                                value="None",
                                label="Color Tone Preset",
                            )
                    with gr.Column(scale=1, elem_classes=["mn-inner-card", "sd-inference-params"]):
                        gr.HTML('<div class="mn-subtitle">Inference Parameters</div>')
                        sd_strength = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            step=0.05,
                            value=0.35,
                            label="Strength",
                        )
                        with gr.Row(elem_classes=["inference-params-row"]):
                            sd_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                step=5,
                                value=10,
                                label="Sampling Steps",
                            )
                            sd_guidance = gr.Slider(
                                minimum=5.0,
                                maximum=15.0,
                                step=0.5,
                                value=7.0,
                                label="Guidance Scale",
                            )

            with gr.Column(elem_classes=["mn-section"]):
                gr.HTML('<div class="mn-section-title">Stable Diffusion Img2Img</div>')
                with gr.Row(elem_classes=["mn-two-col"]):
                    with gr.Column(scale=1, elem_classes=["mn-controls"]):
                        gr.HTML('<div class="mn-subtitle">Content Image</div>')
                        content_sd = gr.Image(
                            type="pil",
                            show_label=False,
                            container=False,
                            height=240,
                            elem_classes=["mn-image"],
                        )
                        gr.HTML('<div class="mn-subtitle">Style Image (Optional)</div>')
                        style_sd = gr.Image(
                            type="filepath",
                            show_label=False,
                            container=False,
                            height=240,
                            elem_classes=["mn-image"],
                        )
                        btn_sd = gr.Button(
                            "Run SD Img2Img",
                            variant="primary",
                            elem_classes=["mn-primary-btn"],
                        )
                    with gr.Column(scale=1, elem_classes=["mn-controls"]):
                        gr.HTML('<div class="mn-subtitle">Result</div>')
                        out_sd = gr.Image(
                            show_label=False,
                            container=False,
                            interactive=False,
                            height=380,
                            elem_classes=["mn-result"],
                        )

            btn_sd.click(
                run_sd_only_ui,
                inputs=[
                    content_sd,
                    sd_prompt_preset,
                    sd_prompt,
                    sd_strength,
                    sd_steps,
                    sd_guidance,
                    sd_scene_preset,
                    sd_color_tone,
                    style_sd,
                    gr.State([]),
                    gr.State({}),
                    gr.State(None),
                ],
                outputs=[out_sd],
            )


if __name__ == "__main__":
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    server_name = os.getenv("CLOTH_SERVER_NAME", "127.0.0.1").strip() or "127.0.0.1"
    server_port = int(os.getenv("CLOTH_SERVER_PORT", "7860").strip() or "7860")
    share = _env_bool("CLOTH_SHARE", default=False)
    auth_user = (os.getenv("CLOTH_AUTH_USER", "") or "").strip()
    auth_password = os.getenv("CLOTH_AUTH_PASSWORD", "") or ""
    auth = (auth_user, auth_password) if auth_user and auth_password else None

    print_server_access_info(
        server_name=server_name,
        server_port=server_port,
        share=share,
        auth_enabled=bool(auth),
    )

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        auth=auth,
        inbrowser=True,
    )
