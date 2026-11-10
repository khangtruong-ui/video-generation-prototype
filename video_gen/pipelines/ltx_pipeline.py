"""LTX-Video DiT pipeline for high-framerate, prompt-conditioned video generation."""

import gc
import logging
from typing import Any, Callable, List, Optional
from PIL import Image
import torch

from video_gen.config import VideoConfig
from video_gen.pipelines.base import BaseVideoPipeline, ProgressCallback

logger = logging.getLogger(__name__)


class LTXVideoPipeline(BaseVideoPipeline):
    """Next-generation Diffusion Transformer (DiT) video pipeline by Lightricks.

    Features:
    - DiT transformer architecture
    - High temporal quality & native 24-25 fps playback
    - Accepts both rich text prompts and conditioning images
    - Fits in ~8-14 GB VRAM on NVIDIA L4 with bfloat16 and model offloading
    """

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-Video",
        config: Optional[VideoConfig] = None,
    ):
        super().__init__(name="ltx_video", device="cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.config = config or VideoConfig.for_l4_gpu()
        self.pipe: Optional[Any] = None

    def load(self) -> None:
        """Load LTX-Video pipeline into memory."""
        if self.pipe is not None:
            return

        from diffusers import LTXImageToVideoPipeline

        logger.info("Loading LTXImageToVideoPipeline from %s...", self.model_id)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        )

        if torch.cuda.is_available():
            # Enable model CPU offload for optimal memory footprint on L4
            self.pipe.enable_model_cpu_offload()
            logger.info("LTX-Video pipeline loaded with model CPU offload.")
        else:
            self.pipe.to("cpu")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        """Unload LTX-Video pipeline and free VRAM."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Unloaded LTX-Video pipeline.")

    def generate_chunk(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        num_frames: int = 25,
        fps: int = 25,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Generate high-resolution video chunk using LTX-Video DiT."""
        if not self.is_loaded():
            self.load()

        assert self.pipe is not None, "Pipeline not loaded"

        # LTX expects dimensions divisible by 32
        # Target ~704x480 or 768x512
        w, h = 704, 480
        if image.size != (w, h):
            image = image.resize((w, h), resample=Image.Resampling.LANCZOS)

        # LTX frames are typically (8 * k + 1) e.g. 17, 25, 33, 49
        ltx_frames = num_frames if (num_frames - 1) % 8 == 0 else ((num_frames // 8) * 8 + 1)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        prompt_text = prompt or "Cinematic smooth motion, photorealistic 8k, seamless camera motion"

        logger.info(
            "Generating LTX chunk: %dx%d, %d frames, steps=%d, prompt='%s'",
            w, h, ltx_frames, num_inference_steps, prompt_text[:60]
        )

        output = self.pipe(
            image=image,
            prompt=prompt_text,
            height=h,
            width=w,
            num_frames=ltx_frames,
            frame_rate=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

        frames = output.frames[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames
