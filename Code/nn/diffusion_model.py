import gc
import os
import traceback
from typing import Optional, Tuple

try:
    from config import (
        AUTO_DOWNLOAD_MODELS,
        DEFAULT_HF_HUB_CACHE,
        DEFAULT_SD_INPAINT_MODEL_ID,
        DEFAULT_SD_MODEL_ID,
        LOCAL_FILES_ONLY,
        AUTO_UNLOAD_DIFFUSION_AFTER_RUN,
        PIPE_OFFLOAD_MODE,
        SD_INPAINT_MODEL_VARIANT,
        SD_MODEL_VARIANT,
        USE_SAFETENSORS,
        clear_cuda_memory,
        resolve_device,
        torch_dtype_for_device,
    )
except ModuleNotFoundError:
    from .config import (
        AUTO_DOWNLOAD_MODELS,
        DEFAULT_HF_HUB_CACHE,
        DEFAULT_SD_INPAINT_MODEL_ID,
        DEFAULT_SD_MODEL_ID,
        LOCAL_FILES_ONLY,
        AUTO_UNLOAD_DIFFUSION_AFTER_RUN,
        PIPE_OFFLOAD_MODE,
        SD_INPAINT_MODEL_VARIANT,
        SD_MODEL_VARIANT,
        USE_SAFETENSORS,
        clear_cuda_memory,
        resolve_device,
        torch_dtype_for_device,
    )
import numpy as np
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter

GARMENT_EDIT_KEYWORDS = ("clothing", "fashion", "garment", "background")


def build_prompts(
    base_prompt: str,
    scene_preset: Optional[str] = None,
    color_tone: Optional[str] = None,
):
    extras = []
    negatives = []

    if scene_preset:
        sp = scene_preset.lower()

        if "clothing" in sp or "fashion" in sp:
            extras.append(
                "fashion product photo, focus on clothes, detailed garment texture, studio lighting"
            )
            negatives.extend(
                [
                    "no background scenery",
                    "no extra objects",
                    "no changing the face",
                ]
            )
        elif "portrait" in sp:
            extras.append(
                "portrait photo, sharp face details, blurred background, subject centered"
            )
            negatives.extend(
                [
                    "no heavy background changes",
                    "no extra logos or text",
                ]
            )
        elif "landscape" in sp:
            extras.append("wide landscape view, detailed environment, natural lighting")
            negatives.extend(["no close-up portrait", "no product shot"])
        elif "city" in sp or "street" in sp:
            extras.append("city street scene, buildings and lights in the background")
            negatives.extend(["no studio product shot"])

    if color_tone:
        ct = color_tone.lower()

        if "warm" in ct:
            extras.append(
                "warm color palette, golden lighting, soft orange and red tones"
            )
            negatives.append("no cold blue tones, no greenish tint")
        elif "cool" in ct:
            extras.append("cool color palette, blue and cyan tones, soft lighting")
            negatives.append("no strong warm orange or red tones")
        elif "pastel" in ct:
            extras.append("soft pastel colors, low contrast, gentle tones")
            negatives.append("no very saturated colors, no harsh contrast")
        elif "high contrast" in ct:
            extras.append("high contrast lighting, strong shadows, dramatic atmosphere")
            negatives.append("no flat lighting, no low contrast")

    positive_prompt = base_prompt.strip()
    if extras:
        if positive_prompt:
            positive_prompt += ", "
        positive_prompt += ", ".join(extras)

    negative_prompt = ", ".join(negatives) if negatives else None
    return positive_prompt, negative_prompt


def soft_blend_style_into_roi(
    base_img: Image.Image,
    style_img: Image.Image,
    roi_box: Tuple[int, int, int, int],
    alpha: float = 0.35,
):
    left, upper, right, lower = roi_box
    roi_w = right - left
    roi_h = lower - upper

    sw, sh = style_img.size
    side = min(sw, sh)
    sx0 = (sw - side) // 2
    sy0 = (sh - side) // 2
    style_crop = style_img.crop((sx0, sy0, sx0 + side, sy0 + side))
    style_resized = style_crop.resize((roi_w, roi_h), Image.LANCZOS)
    style_resized = style_resized.filter(ImageFilter.GaussianBlur(radius=1.2))

    base_roi = base_img.crop(roi_box)
    blended = Image.blend(base_roi, style_resized, alpha=alpha)
    base_img.paste(blended, roi_box)
    return base_img


def load_binary_mask(mask_path: str, target_size=None) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    if target_size is not None:
        mask = mask.resize(target_size, Image.NEAREST)
    return mask


def get_bbox_from_mask(mask_img: Image.Image, pad: int = 8):
    mask_np = np.array(mask_img)
    ys, xs = np.where(mask_np > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    left = max(0, int(xs.min()) - pad)
    right = min(mask_np.shape[1], int(xs.max()) + pad)
    upper = max(0, int(ys.min()) - pad)
    lower = min(mask_np.shape[0], int(ys.max()) + pad)
    return (left, upper, right, lower)


def composite_with_mask(
    original_img: Image.Image,
    edited_img: Image.Image,
    mask_img: Image.Image,
    feather_radius: float = 2.0,
) -> Image.Image:
    mask = mask_img.convert("L")
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return Image.composite(edited_img.convert("RGB"), original_img.convert("RGB"), mask)


class DiffusionRefiner:
    def __init__(
        self,
        model_id: str = DEFAULT_SD_MODEL_ID,
        inpaint_model_id: str = DEFAULT_SD_INPAINT_MODEL_ID,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.inpaint_model_id = inpaint_model_id
        self.device = resolve_device(device)
        self.cache_dir = cache_dir or str(DEFAULT_HF_HUB_CACHE)
        self.local_files_only = LOCAL_FILES_ONLY
        self.auto_download_models = AUTO_DOWNLOAD_MODELS
        self.auto_unload_after_run = AUTO_UNLOAD_DIFFUSION_AFTER_RUN
        self.pipe_offload_mode = PIPE_OFFLOAD_MODE
        self.pipe = None
        self.inpaint_pipe = None

        runtime_dtype = torch_dtype_for_device(self.device)

        print(f"[Diffusion] DiffusionRefiner initialized without loading pipelines")
        print(
            f"[Diffusion] init config | device={self.device}, runtime_dtype={runtime_dtype}"
        )
        print(f"[Diffusion] HF_HOME      = {os.environ.get('HF_HOME')}")
        print(f"[Diffusion] HF_HUB_CACHE = {os.environ.get('HF_HUB_CACHE')}")
        print(f"[Diffusion] cache_dir    = {self.cache_dir}")
        print(f"[Diffusion] DEFAULT_SD_MODEL_ID = {self.model_id}")
        print(f"[Diffusion] DEFAULT_SD_INPAINT_MODEL_ID = {self.inpaint_model_id}")
        print(f"[Diffusion] USE_SAFETENSORS = {USE_SAFETENSORS}")
        print(f"[Diffusion] SD_MODEL_VARIANT = {SD_MODEL_VARIANT!r}")
        print(f"[Diffusion] SD_INPAINT_MODEL_VARIANT = {SD_INPAINT_MODEL_VARIANT!r}")
        print(f"[Diffusion] selected offload mode = {self.pipe_offload_mode!r}")
        print(f"[Diffusion] local_files_only = {self.local_files_only}")
        print(f"[Diffusion] auto_download_models = {self.auto_download_models}")
        print(
            "[Diffusion] auto_unload_after_run = "
            f"{self.auto_unload_after_run}"
        )

    def log_cuda_memory(self, stage: str):
        self._log_cuda_memory(stage)

    def clear_cuda_memory(self):
        clear_cuda_memory()
        gc.collect()
        self._log_cuda_memory("after clear_cuda_memory")

    def unload_img2img_pipe(self):
        print("[Diffusion] before unloading img2img")
        pipe = self.pipe
        self.pipe = None
        if pipe is not None:
            try:
                if hasattr(pipe, "maybe_free_model_hooks"):
                    pipe.maybe_free_model_hooks()
                print(
                    "[Diffusion] deleting img2img pipeline without moving fp16 pipeline to CPU"
                )
            except Exception as exc:
                print(f"[Diffusion][Warning] img2img unload hook cleanup failed: {exc}")
                traceback.print_exc(limit=4)
            try:
                del pipe
            except Exception as exc:
                print(f"[Diffusion][Warning] img2img pipeline delete failed: {exc}")
                traceback.print_exc(limit=4)
        gc.collect()
        clear_cuda_memory()
        self._log_cuda_memory("after unload img2img")

    def unload_inpaint_pipe(self):
        print("[Diffusion] before unloading inpaint")
        pipe = self.inpaint_pipe
        self.inpaint_pipe = None
        if pipe is not None:
            try:
                if hasattr(pipe, "maybe_free_model_hooks"):
                    pipe.maybe_free_model_hooks()
                print(
                    "[Diffusion] deleting inpaint pipeline without moving fp16 pipeline to CPU"
                )
            except Exception as exc:
                print(f"[Diffusion][Warning] inpaint unload hook cleanup failed: {exc}")
                traceback.print_exc(limit=4)
            try:
                del pipe
            except Exception as exc:
                print(f"[Diffusion][Warning] inpaint pipeline delete failed: {exc}")
                traceback.print_exc(limit=4)
        gc.collect()
        clear_cuda_memory()
        self._log_cuda_memory("after unload inpaint")

    def unload_all_pipes(self):
        self.unload_img2img_pipe()
        self.unload_inpaint_pipe()

    def _log_cuda_memory(self, label: str):
        if not torch.cuda.is_available():
            return
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            print(
                f"[Diffusion] {label} GPU free/total GB = "
                f"{free_gb:.2f} / {total_gb:.2f}"
            )
        except Exception as exc:
            print(f"[Diffusion] failed to read CUDA memory info for '{label}': {exc}")

    def _get_img2img_pipe(self):
        if self.pipe is not None:
            return self.pipe

        runtime_dtype = torch_dtype_for_device(self.device)
        load_dtype = torch.float16
        self.unload_inpaint_pipe()
        print("[Diffusion] entering img2img branch, loading img2img pipe")
        self._log_cuda_memory("before loading img2img")
        print(
            f"[Diffusion] loading img2img '{self.model_id}' on device={self.device}, "
            f"runtime_dtype={runtime_dtype}, load_dtype={load_dtype}"
        )
        print(f"[Diffusion] SD_MODEL_VARIANT = {SD_MODEL_VARIANT!r}")

        try:
            img2img_kwargs = {
                "torch_dtype": load_dtype,
                "cache_dir": self.cache_dir,
                "use_safetensors": USE_SAFETENSORS,
                "local_files_only": self.local_files_only and (not self.auto_download_models),
            }
            if SD_MODEL_VARIANT:
                img2img_kwargs["variant"] = SD_MODEL_VARIANT
            print(f"[Diffusion] img2img from_pretrained kwargs = {img2img_kwargs}")

            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                **img2img_kwargs,
            )
            self.pipe = self._configure_pipe(self.pipe, "img2img pipeline")
            self._log_cuda_memory("after loading img2img")
            return self.pipe
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda_memory()
            print(f"[Diffusion] OOM fallback/error message | {exc}")
            raise RuntimeError(
                f"CUDA out of memory while loading diffusion model '{self.model_id}'. "
                "Try closing other GPU tasks, reducing memory usage, or forcing CPU "
                "with CLOTH_DEVICE=cpu."
            ) from exc
        except Exception as exc:
            print(f"[Diffusion] failed to load img2img pipeline: {exc}")
            print(f"[Diffusion] img2img load params | model_id={self.model_id}, kwargs={img2img_kwargs}")
            traceback.print_exc(limit=4)
            raise RuntimeError(
                f"Failed to load img2img model '{self.model_id}'. "
                "If the model cache is missing, run 'python scripts/download_models.py' "
                "or set CLOTH_AUTO_DOWNLOAD_MODELS=true / CLOTH_LOCAL_FILES_ONLY=false. "
                f"Current cache_dir='{self.cache_dir}'."
            ) from exc

    def _configure_pipe(self, pipe, pipe_name: str):
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        if self.device != "cuda":
            pipe = pipe.to(self.device)
            return pipe

        offload_mode = self.pipe_offload_mode
        if offload_mode not in {"cuda", "cpu_offload", "auto"}:
            print(
                f"[Diffusion] unsupported offload mode {offload_mode!r}, fallback to 'cuda'"
            )
            offload_mode = "cuda"

        if pipe_name == "img2img pipeline" and offload_mode == "auto":
            offload_mode = "cuda"

        print(
            f"[Diffusion] {pipe_name} selected offload mode = {offload_mode!r}"
        )

        if offload_mode == "cpu_offload":
            try:
                pipe.enable_model_cpu_offload()
                print(f"[Diffusion] {pipe_name} using enable_model_cpu_offload()")
            except Exception as exc:
                print(
                    f"[Diffusion] {pipe_name} cpu_offload not available, fallback to .to(cuda):",
                    exc,
                )
                pipe = pipe.to(self.device)
                print(f"[Diffusion] {pipe_name} fallback using pipe.to(cuda)")
            return pipe

        pipe = pipe.to(self.device)
        print(f"[Diffusion] {pipe_name} using pipe.to(cuda)")

        return pipe

    def _get_inpaint_pipe(self):
        if self.inpaint_pipe is not None:
            return self.inpaint_pipe

        runtime_dtype = torch_dtype_for_device(self.device)
        load_dtype = torch.float16
        self.unload_img2img_pipe()
        print("[Diffusion] loading inpaint pipe")
        self._log_cuda_memory("before loading inpaint")
        print(
            f"[Diffusion] loading inpaint '{self.inpaint_model_id}' "
            f"on device={self.device}, runtime_dtype={runtime_dtype}, load_dtype={load_dtype}"
        )
        print(f"[Diffusion] inpaint USE_SAFETENSORS = {USE_SAFETENSORS}")
        print(f"[Diffusion] inpaint SD_MODEL_VARIANT = {SD_MODEL_VARIANT!r}")
        print(
            f"[Diffusion] inpaint SD_INPAINT_MODEL_VARIANT = "
            f"{SD_INPAINT_MODEL_VARIANT!r}"
        )
        print(f"[Diffusion] inpaint local_files_only = {self.local_files_only}")

        try:
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.inpaint_model_id,
                torch_dtype=load_dtype,
                cache_dir=self.cache_dir,
                variant=SD_INPAINT_MODEL_VARIANT,
                use_safetensors=USE_SAFETENSORS,
                local_files_only=self.local_files_only and (not self.auto_download_models),
                safety_checker=None,
                requires_safety_checker=False,
            )
            self.inpaint_pipe = self._configure_pipe(
                self.inpaint_pipe, "inpaint pipeline"
            )
            self._log_cuda_memory("after loading inpaint")
            return self.inpaint_pipe
        except torch.cuda.OutOfMemoryError as exc:
            clear_cuda_memory()
            print(f"[Diffusion] OOM fallback/error message | {exc}")
            raise RuntimeError(
                f"CUDA out of memory while loading inpaint model '{self.inpaint_model_id}'. "
                "Try closing other GPU tasks, reducing memory usage, or forcing CPU "
                "with CLOTH_DEVICE=cpu."
            ) from exc
        except Exception as exc:
            print(f"[Diffusion] failed to load inpaint pipeline: {exc}")
            traceback.print_exc(limit=4)
            raise RuntimeError(
                f"Failed to load inpaint model '{self.inpaint_model_id}'. "
                "If the model cache is missing, run 'python scripts/download_models.py' "
                "or set CLOTH_AUTO_DOWNLOAD_MODELS=true / CLOTH_LOCAL_FILES_ONLY=false. "
                f"Current cache_dir='{self.cache_dir}'."
            ) from exc

    def _get_inpaint_branch_label(
        self,
        scene_preset: Optional[str] = None,
        edit_category: Optional[str] = None,
    ) -> str:
        category_text = " ".join(
            part for part in [edit_category, scene_preset] if part
        ).lower()
        if "background" in category_text:
            return "background"
        return "garment"

    def _load_init_image(
        self,
        init_image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
    ) -> Image.Image:
        if init_image is not None:
            return init_image.convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
        raise ValueError("Either init_image or image_path must be provided.")

    def _is_garment_edit(
        self,
        scene_preset: Optional[str] = None,
        edit_category: Optional[str] = None,
    ) -> bool:
        category_text = " ".join(
            part for part in [edit_category, scene_preset] if part
        ).lower()
        return any(keyword in category_text for keyword in GARMENT_EDIT_KEYWORDS)

    def _prepare_garment_mask(
        self,
        garment_mask_path: Optional[str],
        source_size: Tuple[int, int],
        pipeline_size: Tuple[int, int],
    ):
        mask_info = {
            "path": garment_mask_path,
            "exists": False,
            "source_size": source_size,
            "pipeline_size": pipeline_size,
            "mask_original_size": None,
            "mask_pipeline_size": None,
            "is_empty": None,
            "valid": False,
            "reason": None,
        }

        if not garment_mask_path:
            mask_info["reason"] = "missing mask"
            return None, None, mask_info

        if not os.path.isfile(garment_mask_path):
            mask_info["reason"] = "missing mask"
            return None, None, mask_info

        mask_info["exists"] = True

        try:
            mask_original = Image.open(garment_mask_path).convert("L")
        except Exception as exc:
            mask_info["reason"] = f"invalid mask ({exc})"
            return None, None, mask_info

        mask_info["mask_original_size"] = mask_original.size

        if mask_original.size != source_size:
            mask_original = mask_original.resize(source_size, Image.NEAREST)
            mask_info["mask_original_size"] = mask_original.size

        mask_np = np.array(mask_original)
        mask_info["is_empty"] = not np.any(mask_np > 0)
        if mask_info["is_empty"]:
            mask_info["reason"] = "empty mask"
            mask_info["mask_pipeline_size"] = (
                pipeline_size if source_size != pipeline_size else mask_original.size
            )
            return None, None, mask_info

        mask_pipeline = mask_original
        if mask_original.size != pipeline_size:
            mask_pipeline = mask_original.resize(pipeline_size, Image.NEAREST)
        mask_info["mask_pipeline_size"] = mask_pipeline.size

        if not np.any(np.array(mask_pipeline) > 0):
            mask_info["is_empty"] = True
            mask_info["reason"] = "empty mask"
            return None, None, mask_info

        mask_info["valid"] = True
        mask_info["reason"] = "valid mask"
        return mask_original, mask_pipeline, mask_info

    def _build_callback_args(
        self,
        progress_callback,
        num_inference_steps: int,
        strength: Optional[float] = None,
    ):
        if not progress_callback:
            return {}

        callback_steps = 1
        sampling_max_ratio = 0.99
        total_steps = max(1, num_inference_steps)

        def diffusers_callback(step, timestep, latents):
            if strength is None:
                active_steps = max(1, total_steps)
            else:
                active_steps = max(2, int(total_steps * strength))

            if step < active_steps:
                denom = max(1, active_steps - 1)
                sampling_p = step / denom
                total_p = sampling_p * sampling_max_ratio
            else:
                total_p = sampling_max_ratio

            progress_callback(min(total_p, sampling_max_ratio))
            return False

        return {
            "callback": diffusers_callback,
            "callback_steps": callback_steps,
        }

    @torch.no_grad()
    def refine(
        self,
        prompt: str,
        init_image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        strength: float = 0.4,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        keep_resolution: bool = True,
        style_image_path: Optional[str] = None,
        garment_mask_path: Optional[str] = None,
        scene_preset: Optional[str] = None,
        edit_category: Optional[str] = None,
        color_tone: Optional[str] = None,
        blend_style_in_roi: bool = False,
        progress_callback=None,
    ):
        source_image = self._load_init_image(init_image=init_image, image_path=image_path)
        orig_size = source_image.size
        init_resized = source_image.resize((512, 512), Image.LANCZOS)
        use_garment_inpaint = False
        original_mask = None
        pipeline_mask = None
        fallback_reason = None

        print(
            "[Diffusion] refine start | "
            f"scene_preset={scene_preset!r}, edit_category={edit_category!r}, "
            f"garment_mask_path={garment_mask_path!r}"
        )
        print(
            "[Diffusion] image sizes | "
            f"source_size={orig_size}, pipeline_input_size={init_resized.size}"
        )
        print(
            "[Diffusion] garment mask exists | "
            f"{bool(garment_mask_path and os.path.isfile(garment_mask_path))}"
        )

        if blend_style_in_roi and style_image_path is not None and scene_preset:
            try:
                sp = scene_preset.lower()
                if ("clothing" in sp) or ("fashion" in sp):
                    style_img = Image.open(style_image_path).convert("RGB")
                    roi_box = None

                    if garment_mask_path is not None:
                        try:
                            mask_img = load_binary_mask(
                                garment_mask_path, target_size=init_resized.size
                            )
                            roi_box = get_bbox_from_mask(mask_img, pad=10)
                            print(f"[Diffusion] use garment mask ROI: {roi_box}")
                        except Exception as exc:
                            print("[Diffusion] load garment mask failed:", exc)
                            roi_box = None

                    if roi_box is None:
                        width, height = init_resized.size
                        roi_w = int(width * 0.6)
                        roi_h = int(height * 0.45)
                        left = (width - roi_w) // 2
                        right = left + roi_w
                        upper = int(height * 0.35)
                        lower = upper + roi_h
                        roi_box = (left, upper, right, lower)
                        print(f"[Diffusion] fallback ROI: {roi_box}")

                    init_resized = soft_blend_style_into_roi(
                        base_img=init_resized,
                        style_img=style_img,
                        roi_box=roi_box,
                        alpha=0.35,
                    )
            except Exception as exc:
                print("[Diffusion] soft_blend_style_in_roi failed:", exc)

        positive_prompt, negative_prompt = build_prompts(
            base_prompt=prompt,
            scene_preset=scene_preset,
            color_tone=color_tone,
        )

        is_garment_scene = self._is_garment_edit(
            scene_preset=scene_preset, edit_category=edit_category
        )
        print(f"[Diffusion] garment scene detected | {is_garment_scene}")

        if is_garment_scene:
            original_mask, pipeline_mask, mask_info = self._prepare_garment_mask(
                garment_mask_path=garment_mask_path,
                source_size=orig_size,
                pipeline_size=init_resized.size,
            )
            print(
                "[Diffusion] mask info | "
                f"path={mask_info['path']!r}, exists={mask_info['exists']}, "
                f"mask_original_size={mask_info['mask_original_size']}, "
                f"mask_pipeline_size={mask_info['mask_pipeline_size']}, "
                f"source_size={mask_info['source_size']}, "
                f"pipeline_input_size={mask_info['pipeline_size']}, "
                f"is_empty={mask_info['is_empty']}, valid={mask_info['valid']}"
            )
            if pipeline_mask is not None:
                use_garment_inpaint = True
                print("[Diffusion] inpaint triggered | True")
            else:
                fallback_reason = mask_info["reason"] or "invalid mask"
                print("[Diffusion] inpaint triggered | False")
                print(f"[Diffusion] fallback reason | {fallback_reason}")
        else:
            fallback_reason = "non-garment scene"
            print("[Diffusion] inpaint triggered | False")
            print(f"[Diffusion] fallback reason | {fallback_reason}")

        callback_args = self._build_callback_args(
            progress_callback=progress_callback,
            num_inference_steps=num_inference_steps,
            strength=strength,
        )
        active_pipeline_type = "inpaint" if use_garment_inpaint else "img2img"

        try:
            if use_garment_inpaint:
                branch_label = self._get_inpaint_branch_label(
                    scene_preset=scene_preset,
                    edit_category=edit_category,
                )
                print("[Diffusion] active pipeline | inpaint")
                print(f"[Diffusion] entering {branch_label} inpaint branch")
                pipe = self._get_inpaint_pipe()
                self._log_cuda_memory("before inpaint inference")
                result = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=init_resized,
                    mask_image=pipeline_mask,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    **callback_args,
                )
                self._log_cuda_memory("after inpaint inference")
            else:
                print("[Diffusion] active pipeline | img2img")
                pipe = self._get_img2img_pipe()
                self._log_cuda_memory("before img2img inference")
                result = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=init_resized,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    **callback_args,
                )
                self._log_cuda_memory("after img2img inference")

            if progress_callback:
                progress_callback(0.99)

            out_image = result.images[0]

            if keep_resolution:
                out_image = out_image.resize(orig_size, Image.LANCZOS)
                if use_garment_inpaint and original_mask is not None:
                    out_image = composite_with_mask(source_image, out_image, original_mask)
                    print("[Diffusion] final composite executed | True")
                else:
                    print("[Diffusion] final composite executed | False")
            elif use_garment_inpaint and pipeline_mask is not None:
                out_image = composite_with_mask(init_resized, out_image, pipeline_mask)
                print("[Diffusion] final composite executed | True")
            else:
                print("[Diffusion] final composite executed | False")

            if progress_callback:
                progress_callback(1.0)

            return out_image
        except torch.cuda.OutOfMemoryError as exc:
            self.clear_cuda_memory()
            print(f"[Diffusion] OOM fallback/error message | {exc}")
            raise RuntimeError(
                "GPU memory is not enough for this diffusion task. Try lower steps, lower strength, or restart Gradio."
            ) from exc
        except Exception as exc:
            print(f"[Diffusion] diffusion inference failed: {exc}")
            traceback.print_exc(limit=4)
            self.clear_cuda_memory()
            if "DefaultCPUAllocator: not enough memory" in str(exc):
                raise RuntimeError(
                    "CPU memory is not enough during diffusion offload. Try using direct CUDA mode or restart Gradio."
                ) from exc
            raise RuntimeError(
                f"Diffusion inference failed: {exc}"
            ) from exc
        finally:
            gc.collect()
            clear_cuda_memory()
            if self.auto_unload_after_run:
                if active_pipeline_type == "inpaint":
                    self.unload_inpaint_pipe()
                else:
                    self.unload_img2img_pipe()
