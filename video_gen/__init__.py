"""Long Video Generation Prototype.

Extensible long video generation package using Stable Video Diffusion XT,
autoregressive chunk chaining, and FastAPI service endpoints.
"""

from video_gen.config import DeviceConfig, GenerationConfig, VideoConfig
from video_gen.generator import GenerationResult, LongVideoGenerator
from video_gen.api.app import app, create_app
from video_gen.colab import display_video, download_video, is_colab

__version__ = "0.1.0"

__all__ = [
    "VideoConfig",
    "DeviceConfig",
    "GenerationConfig",
    "LongVideoGenerator",
    "GenerationResult",
    "app",
    "create_app",
    "display_video",
    "download_video",
    "is_colab",
    "__version__",
]
