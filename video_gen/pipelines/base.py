"""Abstract base class for video generation pipelines."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from PIL import Image


ProgressCallback = Callable[[int, int, str], None]


class BaseVideoPipeline(ABC):
    """Abstract interface defining video chunk generation."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """Load model weights and move to destination device."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if pipeline components are currently loaded in memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload pipeline components to free GPU/CPU memory."""
        pass

    @abstractmethod
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
        """Generate a single clip/chunk of video frames conditioned on an input image.

        Args:
            image: Conditioning input image (PIL RGB Image).
            num_frames: Number of output frames (default 25).
            fps: Intended framerate metadata.
            motion_bucket_id: Motion intensity index (1-255).
            noise_aug_strength: Initial noise level added to conditioning image.
            num_inference_steps: Denoising steps.
            min_guidance_scale: Classifier-free guidance min scale.
            max_guidance_scale: Classifier-free guidance max scale.
            seed: Reproducibility seed.
            callback: Optional progress reporter callback(step, total_steps, message).
            **kwargs: Extra model-specific keyword arguments.

        Returns:
            List of generated PIL Images representing consecutive frames.
        """
        pass
