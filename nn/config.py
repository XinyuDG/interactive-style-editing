import os
from pathlib import Path
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


NN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("CLOTH_PROJECT_ROOT", NN_DIR.parent)).resolve()
DEFAULT_HF_HOME = Path(os.getenv("HF_HOME", str(PROJECT_ROOT))).resolve()
DEFAULT_HF_HUB_CACHE = Path(
    os.getenv("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
).resolve()
DEFAULT_RESULT_DIR = Path(
    os.getenv("CLOTH_RESULT_DIR", str(PROJECT_ROOT / "nn" / "result"))
).resolve()
DEFAULT_SD_MODEL_ID = os.getenv(
    "CLOTH_SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
)
DEFAULT_SD_INPAINT_MODEL_ID = os.getenv(
    "CLOTH_SD_INPAINT_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-inpainting"
)
DEFAULT_DEVICE_OVERRIDE = os.getenv("CLOTH_DEVICE", "").strip().lower() or None
DEFAULT_LOCAL_FILES_ONLY = _env_flag("CLOTH_LOCAL_FILES_ONLY", default=True)
AUTO_DOWNLOAD_MODELS = _env_flag("CLOTH_AUTO_DOWNLOAD_MODELS", default=False)
USE_SAFETENSORS = True
SD_MODEL_VARIANT = os.getenv("CLOTH_SD_MODEL_VARIANT", "fp16").strip()
SD_INPAINT_MODEL_VARIANT = os.getenv("CLOTH_SD_INPAINT_MODEL_VARIANT", "fp16").strip()
PIPE_OFFLOAD_MODE = os.getenv("CLOTH_PIPE_OFFLOAD_MODE", "cuda").strip().lower()
MODEL_VARIANT = SD_MODEL_VARIANT

HF_HOME = str(DEFAULT_HF_HOME)
HF_HUB_CACHE = str(DEFAULT_HF_HUB_CACHE)
RESULT_DIR = str(DEFAULT_RESULT_DIR)
LOCAL_FILES_ONLY = DEFAULT_LOCAL_FILES_ONLY
AUTO_UNLOAD_DIFFUSION_AFTER_RUN = _env_flag(
    "CLOTH_AUTO_UNLOAD_DIFFUSION_AFTER_RUN",
    default=True,
)

DEFAULT_HF_HOME.mkdir(parents=True, exist_ok=True)
DEFAULT_HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
DEFAULT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ.setdefault("TORCH_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HUB_CACHE)
if LOCAL_FILES_ONLY:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

print(f"[Config] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[Config] HF_HOME={HF_HOME}")
print(f"[Config] HF_HUB_CACHE={HF_HUB_CACHE}")
print(f"[Config] RESULT_DIR={RESULT_DIR}")
print(f"[Config] DEFAULT_SD_MODEL_ID={DEFAULT_SD_MODEL_ID}")
print(f"[Config] DEFAULT_SD_INPAINT_MODEL_ID={DEFAULT_SD_INPAINT_MODEL_ID}")
print(f"[Config] USE_SAFETENSORS={USE_SAFETENSORS}")
print(f"[Config] SD_MODEL_VARIANT={SD_MODEL_VARIANT!r}")
print(f"[Config] SD_INPAINT_MODEL_VARIANT={SD_INPAINT_MODEL_VARIANT!r}")
print(f"[Config] PIPE_OFFLOAD_MODE={PIPE_OFFLOAD_MODE!r}")
print(f"[Config] LOCAL_FILES_ONLY={LOCAL_FILES_ONLY}")
print(f"[Config] AUTO_DOWNLOAD_MODELS={AUTO_DOWNLOAD_MODELS}")
print("[Config] AUTO_UNLOAD_DIFFUSION_AFTER_RUN=" f"{AUTO_UNLOAD_DIFFUSION_AFTER_RUN}")

import torch


def resolve_device(requested: Optional[str] = None) -> str:
    device = (requested or DEFAULT_DEVICE_OVERRIDE or "cuda").lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device '{device}'. Expected 'cuda' or 'cpu'.")

    if device == "cuda" and not torch.cuda.is_available():
        print("[Config] CUDA not available, falling back to CPU.")
        return "cpu"

    return device


def torch_dtype_for_device(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
