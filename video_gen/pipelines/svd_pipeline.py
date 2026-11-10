"""Stable Video Diffusion (SVD-XT) pipeline optimized for NVIDIA L4 GPU."""

import gc
import logging
from typing import Any, Callable, List, Optional
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline

from video_gen.config import VideoConfig
from video_gen.pipelines.base import BaseVideoPipeline, ProgressCallback

logger = logging.getLogger(__name__)


class SVDPipeline(BaseVideoPipeline):
    """Production SVD-XT pipeline optimized for NVIDIA L4 (24GB VRAM)."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        config: Optional[VideoConfig] = None,
    ):
        super().__init__(name="svd_xt", device="cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.config = config or VideoConfig.for_l4_gpu()
        self.pipe: Optional[StableVideoDiffusionPipeline] = None

    def load(self) -> None:
        """Load SVD-XT pipeline into GPU memory with L4 optimizations."""
        if self.pipe is not None:
            logger.info("SVDPipeline is already loaded.")
            return

        device_cfg = self.config.device
        dtype = device_cfg.get_torch_dtype()

        logger.info(
            "Loading StableVideoDiffusionPipeline (%s) onto device=%s (dtype=%s)...",
            self.model_id,
            device_cfg.device,
            device_cfg.torch_dtype,
        )

        try:
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
        except TypeError:
            # Fallback for newer diffusers expecting dtype
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                dtype=dtype,
            )

        # Move to GPU or enable cpu offload
        if device_cfg.device.startswith("cuda") and torch.cuda.is_available():
            if device_cfg.enable_model_cpu_offload:
                logger.info("Enabling model CPU offload...")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(device_cfg.device)

            # Enable VAE / attention memory optimizations if supported
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()

            logger.info("SVDPipeline successfully loaded and configured for L4 GPU.")
        else:
            logger.warning("CUDA is not available. SVDPipeline running on CPU (will be slow).")
            self.pipe.to("cpu")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        """Unload pipeline and release GPU VRAM."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            logger.info("Unloaded SVDPipeline and freed GPU memory.")

    def generate_chunk(
        self,
        image: Image.Image,
        num_frames: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Generate a 25-frame clip from conditioning image using SVD-XT."""
        if not self.is_loaded():
            self.load()

        assert self.pipe is not None, "Pipeline was not initialized"

        width, height = self.config.generation.width, self.config.generation.height

        # Ensure image matches target aspect ratio/dimensions
        if image.size != (width, height):
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

        # Generator for deterministic results if seed is set
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        decode_chunk_size = self.config.device.decode_chunk_size

        logger.info(
            "Generating SVD chunk: %dx%d, %d frames @ %dfps, motion_bucket=%d, steps=%d",
            width,
            height,
            num_frames,
            fps,
            motion_bucket_id,
            num_inference_steps,
        )

        step_counter = 0

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            nonlocal step_counter
            step_counter = step_index + 1
            if callback:
                callback(step_counter, num_inference_steps, f"Denoising step {step_counter}/{num_inference_steps}")
            return callback_kwargs

        call_kwargs = dict(
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            generator=generator,
            output_type="pil",
        )

        # Diffusers step callback
        try:
            call_kwargs["callback_on_step_end"] = step_callback
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            output = self.pipe(**call_kwargs)
        except TypeError:
            # Fallback if callback signature differs
            call_kwargs.pop("callback_on_step_end", None)
            call_kwargs.pop("callback_on_step_end_tensor_inputs", None)
            output = self.pipe(**call_kwargs)

        frames = output.frames[0]

        # Free intermediate cached tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Generated %d frames successfully.", len(frames))
        return frames
