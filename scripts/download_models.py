import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from torchvision.models import VGG19_Weights, vgg19


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_HOME = Path(os.getenv("HF_HOME", str(PROJECT_ROOT))).resolve()
HF_HUB_CACHE = Path(
    os.getenv("HF_HUB_CACHE", str(PROJECT_ROOT / "hub"))
).resolve()
HF_TOKEN = (os.getenv("HF_TOKEN", "") or "").strip() or None

COMMON_IGNORE_PATTERNS = [
    "*.bin",
    "*.ckpt",
    "*.msgpack",
    "*.onnx",
    "*.pb",
    "*.h5",
    "*.ot",
    "*.pt",
]

SD_FP16_ALLOW_PATTERNS = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "config.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "text_encoder/config.json",
    "text_encoder/*.fp16.safetensors",
    "unet/config.json",
    "unet/*.fp16.safetensors",
    "vae/config.json",
    "vae/*.fp16.safetensors",
    "safety_checker/config.json",
    "safety_checker/*.fp16.safetensors",
    "*.json",
    "*.txt",
    "*.md",
]

MODEL_DOWNLOAD_SPECS = [
    {
        "repo_id": "fashn-ai/fashn-human-parser",
        "strategy": "full snapshot",
        "allow_patterns": None,
        "ignore_patterns": None,
    },
    {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "strategy": "fp16 safetensors only",
        "allow_patterns": SD_FP16_ALLOW_PATTERNS,
        "ignore_patterns": COMMON_IGNORE_PATTERNS,
    },
    {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "strategy": "fp16 safetensors only",
        "allow_patterns": SD_FP16_ALLOW_PATTERNS,
        "ignore_patterns": COMMON_IGNORE_PATTERNS,
    },
]


def download_model(spec: dict) -> None:
    repo_id = spec["repo_id"]
    strategy = spec["strategy"]
    print(f"[Download] start {repo_id}")
    print(f"[Download] strategy: {strategy}")

    snapshot_kwargs = {
        "repo_id": repo_id,
        "cache_dir": str(HF_HUB_CACHE),
        "token": HF_TOKEN,
    }
    if spec["allow_patterns"] is not None:
        snapshot_kwargs["allow_patterns"] = spec["allow_patterns"]
    if spec["ignore_patterns"] is not None:
        snapshot_kwargs["ignore_patterns"] = spec["ignore_patterns"]

    local_path = snapshot_download(**snapshot_kwargs)

    print(f"[Download] done {repo_id}")
    print(f"[Download] path: {local_path}")
    print(f"[Download] strategy confirmed: {strategy}")
    if strategy == "fp16 safetensors only":
        print(
            "[Download] If local model loading fails, check whether the repo still "
            "provides the required fp16 safetensors files."
        )


def download_vgg19_weights() -> None:
    print("[Download] start torchvision VGG19 weights")
    try:
        _ = vgg19(weights=VGG19_Weights.DEFAULT)
        print("[Download] done torchvision VGG19 weights")
    except Exception as exc:
        print(f"[Download] failed torchvision VGG19 weights: {exc}")
        print("[Download] VGG19 download failed. Check network or run again.")
        raise


def main() -> int:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))

    print(f"[Download] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[Download] HF_HOME={HF_HOME}")
    print(f"[Download] HF_HUB_CACHE={HF_HUB_CACHE}")

    failed = []
    for spec in MODEL_DOWNLOAD_SPECS:
        try:
            download_model(spec)
        except Exception as exc:
            repo_id = spec["repo_id"]
            print(f"[Download] failed {repo_id}: {exc}")
            print("[Download] Possible causes:")
            print("[Download] 1. Network connection was interrupted.")
            print("[Download] 2. Hugging Face is not reachable from this machine.")
            print("[Download] 3. Authentication is required but HF_TOKEN is not set.")
            print("[Download] You can retry: python scripts/download_models.py")
            print("[Download] Optional faster HF download helper: pip install hf_xet")
            failed.append(repo_id)

    if failed:
        print("[Download] Some model downloads failed.")
        print("[Download] Please fix network or access issues and retry.")
        return 1

    try:
        download_vgg19_weights()
    except Exception:
        return 1

    print("[Download] All required models are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
