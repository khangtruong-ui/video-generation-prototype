"""Wan2.1 Image-to-Video foundation pipeline optimized for NVIDIA L4 GPU."""

import gc
import logging
from typing import Any, Callable, List, Optional
import numpy as np
from PIL import Image
import torch

from video_gen.config import VideoConfig
from video_gen.pipelines.base import BaseVideoPipeline, ProgressCallback

logger = logging.getLogger(__name__)

# Ensure ftfy compatibility for diffusers Wan pipeline
try:
    import ftfy
except ImportError:
    import types
    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda text: text

try:
    import diffusers.pipelines.wan.pipeline_wan_i2v as wan_i2v_mod
    if not hasattr(wan_i2v_mod, "ftfy"):
        wan_i2v_mod.ftfy = ftfy
except Exception:
    pass

try:
    import diffusers.pipelines.wan.pipeline_wan as wan_t2v_mod
    if not hasattr(wan_t2v_mod, "ftfy"):
        wan_t2v_mod.ftfy = ftfy
except Exception:
    pass

DEFAULT_WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background"
)


class WanVideoPipeline(BaseVideoPipeline):
    """Wan2.1 14B Image-to-Video pipeline optimized for NVIDIA L4 GPU (24GB VRAM).

    Key architectural highlights:
    - 14B parameter high-capacity Diffusion Transformer (DiT)
    - Native 16 fps video generation at 480P (832x480) or 720P (1280x720)
    - 81 frames per chunk (~5.06s) with seamless autoregressive conditioning
    - Utilizes bfloat16 precision native to Ada Lovelace / NVIDIA L4 tensor cores
    - Model CPU offload enables loading full 14B weights with optimal throughput on 24GB VRAM
    - 3D VAE slicing & tiling eliminates peak VRAM decoding bottlenecks
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        config: Optional[VideoConfig] = None,
    ):
        super().__init__(name="wan", device="cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.config = config or VideoConfig.for_l4_gpu()
        self.pipe: Optional[Any] = None

    def load(self) -> None:
        """Load Wan2.1 pipeline into GPU memory with L4 optimizations."""
        if self.pipe is not None:
            logger.info("WanVideoPipeline is already loaded.")
            return

        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import CLIPVisionModel

        dtype = self.config.device.get_torch_dtype()

        logger.info(
            "Loading WanImageToVideoPipeline (%s) onto device=%s (dtype=%s)...",
            self.model_id,
            self.config.device.device,
            dtype,
        )

        try:
            # Optimal precision configuration: fp32 for CLIP vision and VAE, bfloat16 for 14B transformer
            image_encoder = CLIPVisionModel.from_pretrained(
                self.model_id,
                subfolder="image_encoder",
                torch_dtype=torch.float32,
            )
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                image_encoder=image_encoder,
                torch_dtype=dtype,
            )
        except Exception as exc:
            logger.info("Direct subfolder loading fallback: %s. Using standard from_pretrained...", exc)
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )

        if self.config.device.device.startswith("cuda") and torch.cuda.is_available():
            # Enable sequential model CPU offload: allows the 14B model to run comfortably on 24GB L4
            if self.config.device.enable_model_cpu_offload:
                logger.info("Enabling model CPU offload for 14B Wan model on L4 GPU...")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to("cuda")

            # Enable 3D VAE memory optimizations
            vae_module = getattr(self.pipe, "vae", None)
            if vae_module is not None:
                if hasattr(vae_module, "enable_slicing") and self.config.device.enable_vae_slicing:
                    vae_module.enable_slicing()
                if hasattr(vae_module, "enable_tiling") and self.config.device.enable_vae_tiling:
                    vae_module.enable_tiling()

            logger.info("WanVideoPipeline successfully loaded and optimized for NVIDIA L4 GPU.")
        else:
            logger.warning("CUDA is not available. WanVideoPipeline running on CPU (will be slow).")
            self.pipe.to("cpu")

    def is_loaded(self) -> bool:
        """Check if pipeline components are currently resident in memory."""
        return self.pipe is not None

    def unload(self) -> None:
        """Release pipeline components and free GPU VRAM."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            logger.info("Unloaded WanVideoPipeline and released GPU memory.")

    def generate_chunk(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        num_frames: int = 81,
        fps: int = 16,
        motion_bucket_id: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        num_inference_steps: int = 30,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 5.0,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Generate video chunk using Wan2.1 foundation DiT model.

        Args:
            image: Conditioning input image (PIL RGB image).
            prompt: Text prompt guiding video motion and details.
            num_frames: Number of output frames (default 81, or 4k+1 e.g. 17, 33, 49, 81).
            fps: Intended framerate metadata (native 16 fps).
            motion_bucket_id: Kept for interface compatibility.
            noise_aug_strength: Kept for interface compatibility.
            num_inference_steps: Denoising diffusion steps (default 30).
            min_guidance_scale: Minimum CFG guidance.
            max_guidance_scale: Maximum CFG guidance.
            guidance_scale: Guidance scale (default 5.0).
            negative_prompt: Optional negative prompt.
            seed: RNG seed for reproducible generation.
            callback: Optional progress reporter callback(step, total_steps, message).
            **kwargs: Extra model-specific keyword arguments.

        Returns:
            List of generated PIL Images.
        """
        if not self.is_loaded():
            self.load()

        assert self.pipe is not None, "Wan pipeline was not initialized"

        width = self.config.generation.width
        height = self.config.generation.height

        # Ensure image matches target aspect ratio / dimensions (must be divisible by 16)
        width = (width // 16) * 16
        height = (height // 16) * 16

        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

        # Wan requires (num_frames - 1) % 4 == 0 (e.g. 17, 33, 49, 65, 81)
        if (num_frames - 1) % 4 != 0:
            adjusted_frames = max(17, ((num_frames - 1) // 4) * 4 + 1)
            logger.info("Adjusted num_frames from %d to %d for Wan architecture (4k+1)", num_frames, adjusted_frames)
            num_frames = adjusted_frames

        # Resolve guidance scale
        scale = guidance_scale if guidance_scale is not None else max_guidance_scale
        if scale is None:
            scale = 5.0

        prompt_text = prompt or "Cinematic high-definition footage, smooth natural camera motion, ultra-detailed photorealistic"
        neg_prompt = negative_prompt or DEFAULT_WAN_NEGATIVE_PROMPT

        # Generator device: with model CPU offload, keep generator on CPU
        generator = None
        if seed is not None:
            gen_device = "cpu" if self.config.device.enable_model_cpu_offload else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        logger.info(
            "Generating Wan2.1 chunk: %dx%d, %d frames @ %dfps, steps=%d, guidance=%.1f",
            width,
            height,
            num_frames,
            fps,
            num_inference_steps,
            scale,
        )

        step_counter = 0

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            nonlocal step_counter
            step_counter = step_index + 1
            if callback:
                callback(step_counter, num_inference_steps, f"Wan denoising step {step_counter}/{num_inference_steps}")
            return callback_kwargs

        call_kwargs = dict(
            image=image,
            prompt=prompt_text,
            negative_prompt=neg_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=scale,
            generator=generator,
            output_type="np",
        )

        try:
            call_kwargs["callback_on_step_end"] = step_callback
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            output = self.pipe(**call_kwargs)
        except TypeError:
            call_kwargs.pop("callback_on_step_end", None)
            call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
            output = self.pipe(**call_kwargs)

        raw_frames = output.frames[0]

        # Convert output frames to PIL RGB Images
        pil_frames: List[Image.Image] = []
        for f in raw_frames:
            if isinstance(f, Image.Image):
                pil_frames.append(f)
            elif isinstance(f, np.ndarray):
                arr = f
                if arr.dtype != np.uint8:
                    max_val = arr.max()
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if max_val <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(arr))
            elif isinstance(f, torch.Tensor):
                arr = f.detach().cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    max_val = arr.max()
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if max_val <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
                pil_frames.append(Image.fromarray(arr))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Wan2.1 chunk generated %d frames successfully.", len(pil_frames))
        return pil_frames
