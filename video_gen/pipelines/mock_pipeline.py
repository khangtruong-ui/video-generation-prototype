"""Fast mock pipeline for testing, CI, and GPU-free debugging."""

import logging
import math
from typing import Any, Callable, List, Optional
import numpy as np
from PIL import Image

from video_gen.pipelines.base import BaseVideoPipeline, ProgressCallback

logger = logging.getLogger(__name__)


class MockVideoPipeline(BaseVideoPipeline):
    """Generates procedural video frames rapidly without loading deep learning weights."""

    def __init__(self, name: str = "mock", device: str = "cpu"):
        super().__init__(name=name, device=device)
        self._is_loaded = False

    def load(self) -> None:
        logger.info("Loaded MockVideoPipeline (synthetic procedural generator)")
        self._is_loaded = True

    def is_loaded(self) -> bool:
        return self._is_loaded

    def unload(self) -> None:
        logger.info("Unloaded MockVideoPipeline")
        self._is_loaded = False

    def generate_chunk(
        self,
        image: Image.Image,
        num_frames: int = 10,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        num_inference_steps: int = 2,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        seed: Optional[int] = None,
        callback: Optional[ProgressCallback] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Generate procedural pan/zoom video frames from source image."""
        if not self._is_loaded:
            self.load()

        orig_w, orig_h = image.size
        frames: List[Image.Image] = []

        # Motion speed derived from motion_bucket_id
        speed_factor = motion_bucket_id / 127.0

        for step in range(num_frames):
            t = step / max(1, num_frames - 1)
            # Subtle smooth sinusoidal zoom (1.0 to 1.05) and pan
            zoom = 1.0 + 0.05 * speed_factor * math.sin(t * math.pi)
            crop_w = int(orig_w / zoom)
            crop_h = int(orig_h / zoom)

            dx = int((orig_w - crop_w) * 0.5 * (1.0 + 0.2 * math.cos(t * math.pi * 2)))
            dy = int((orig_h - crop_h) * 0.5 * (1.0 + 0.2 * math.sin(t * math.pi * 2)))

            dx = max(0, min(dx, orig_w - crop_w))
            dy = max(0, min(dy, orig_h - crop_h))

            cropped = image.crop((dx, dy, dx + crop_w, dy + crop_h))
            frame = cropped.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

            # Add subtle dynamic tint change across frames to simulate lighting
            arr = np.array(frame, dtype=np.float32)
            tint = 1.0 + 0.03 * math.sin(t * 3.1415)
            arr = np.clip(arr * tint, 0, 255).astype(np.uint8)

            frames.append(Image.fromarray(arr))

            if callback:
                callback(step + 1, num_frames, f"Generating mock frame {step + 1}/{num_frames}")

        return frames
